import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict

import pandas as pd
from database_tools.adapters.postgresql import PostgresqlAdapter
from llm_adapters import model_config
from llm_adapters.llm_adapter import LLMClientFactory
from slack_bot.notifications import SlackNotifier
from sqlalchemy import desc

from exchange_adapter import BaseExchangeAdapter
from model.turtle_model import StrategySettings
from src.config import LLM_TRADER_SLACK_URL, VALIDATOR_REPEATED_CALL_TIME_TEST_MIN
from src.model import trader_database
from src.model.turtle_model import AgentActions
from src.prompts import llm_trader_prompt, llm_turtle_validator
from src.utils.utils import calculate_macd, calculate_rsi, \
    calculate_sma, calculate_bollinger_bands, calculate_stochastic_oscillator, round_series, \
    calculate_auto_fibonacci, calculate_pivot_points, calculate_adx, calculate_atr, calculate_obv, \
    shorten_large_numbers, dynamic_safe_round

_logger = logging.getLogger(__name__)
_notifier = SlackNotifier(url=LLM_TRADER_SLACK_URL, username='main')


class ConditionVerificationError(Exception):
    """Problem with a verification before calling the agent"""


@dataclass
class AgentAction:
    action: str = None
    amount: float = None
    entry_price: float = None
    order_id: str = None
    order_type: str = None
    rationale: str = None
    stop_loss: float = None
    take_profit: float = None


class LmmTrader:
    SYSTEM_PROMPT = llm_trader_prompt
    agent_action_obj = AgentAction
    df_tail_for_agent = 20

    def __init__(self,
                 exchange: BaseExchangeAdapter,
                 strategy_settings: StrategySettings = None,
                 db: PostgresqlAdapter = None,
                 load_data=True,
                 ):
        self.strategy_settings = strategy_settings
        self._exchange = exchange
        self._database = trader_database if not db else db

        if load_data:
            self.last_candle_timestamp = ...
            self.last_close_price = ...
            self.price_action_data = self.get_price_action_data()
            self.opened_positions = self.get_open_positions_data()
            self.opened_orders = self.get_open_orders_data()
            self.trade_history = self.get_trade_history_data()
            self.last_agent_output = self.get_last_agent_output()
            self.exchange_settings = self.get_exchange_settings()

            self.llm_input_data = self.create_llm_input_dict()

    @classmethod
    def init_just_exchange(cls,
                           exchange: BaseExchangeAdapter,
                           strategy_settings: StrategySettings = None,
                           db: PostgresqlAdapter = None,
                           ):
        return cls(exchange=exchange, strategy_settings=strategy_settings, db=db, load_data=False)

    def get_last_agent_output(self):
        with self._database.session_manager() as session:
            last_agent_action = (
                session.query(
                    AgentActions.agent_output,
                    AgentActions.timestamp_created,
                    AgentActions.candle_timestamp
                )
                .filter(AgentActions.strategy_id == self.strategy_settings.id)
                .order_by(desc(AgentActions.timestamp_created))
                .first()  # Get the most recent record
            )

        # If there's no record, return None
        if not last_agent_action:
            return None

        # Unpack the tuple and return as a dictionary
        agent_output, timestamp_created, candle_timestamp = last_agent_action
        return {
            'agent_output': agent_output,  # JSON object
            'timestamp': timestamp_created,  # Timestamp
            'candle_timestamp': candle_timestamp
        }

    def get_exchange_settings(self):
        _logger.info("getting exchange settings...")
        free_balance = round(self._exchange.free_balance, 0)
        total_balance = round(self._exchange.total_balance, 0)
        max_amount = dynamic_safe_round(free_balance * 0.9 / self.last_close_price, 2)
        return {
            'min_cost': self._exchange.min_cost,
            'min_amount': self._exchange.min_amount,
            'free_capital': free_balance,
            'total_capital': total_balance,
            'max_amount_based_on_free_capital': max_amount
        }

    def preprocess_open_interest(self):
        oi = self._exchange.get_open_interest_hist(timeframe=self.strategy_settings.timeframe)
        if oi:
            oi_df = pd.DataFrame(oi)
            keep_cols = ['timestamp', 'openInterestValue']
            oi_df['openInterestValue'] = round_series(oi_df['openInterestValue'], 0)
            oi_df = oi_df[keep_cols]
            oi_df['openInterest_sma_20'] = calculate_sma(oi_df, 20, column='openInterestValue')
            oi_df['openInterest_sma_10'] = calculate_sma(oi_df, 10, column='openInterestValue')
            oi_df.set_index('timestamp', inplace=True)
            return oi_df
        else:
            return None

    def preprocess_funding_rate(self):
        fr = self._exchange.get_funding_rate_history()  # TODO: resample timeframe when 1d
        if fr:
            fr_df = pd.DataFrame(fr)
            keep_cols = ['timestamp', 'fundingRate']
            fr_df['fundingRate'] = round_series(fr_df['fundingRate'], 2)
            fr_df = fr_df[keep_cols]
            fr_df.set_index('timestamp', inplace=True)
            return fr_df
        else:
            return None

    def get_ohcl_data(self):
        ohlc = self._exchange.fetch_ohlc(
            days=self.strategy_settings.buffer_days, timeframe=self.strategy_settings.timeframe
        )
        return ohlc

    def get_open_positions_data(self):
        op = self._exchange.get_opened_position()[0]
        contracts = op['contracts']
        if contracts > 0:
            return {
                'timestamp': op.get('timestamp', None),
                'datetime': op.get('datetime', None),
                'initialMargin': op.get('initialMargin', None),
                'initialMarginPercentage': op.get('initialMarginPercentage', None),
                'maintenanceMargin': op.get('maintenanceMargin', None),
                'maintenanceMarginPercentage': op.get('maintenanceMarginPercentage', None),
                'entryPrice': op.get('entryPrice', None),
                'notional': op.get('notional', None),
                'leverage': op.get('leverage', None),
                'unrealizedPnl': op.get('unrealizedPnl', None),
                'contracts': op.get('contracts', None),
                'contractSize': op.get('contractSize', None),
                'marginRatio': op.get('marginRatio', None),
                'liquidationPrice': op.get('liquidationPrice', None),
                'markPrice': op.get('markPrice', None),
                'collateral': op.get('collateral', None),
                'marginMode': op.get('marginMode', None),
                'side': op.get('side', None),
                'percentage': op.get('percentage', None),
                'stopLossPrice': op.get('stopLossPrice', None),
                'takeProfitPrice': op.get('takeProfitPrice', None)
            }
        else:
            return None

    def get_open_orders_data(self):
        open_orders = self._exchange.get_opened_orders()  # Fetch all open orders
        if open_orders:
            open_orders_list = []
            for order in open_orders:
                open_orders_list.append({
                    'amount': order.get('amount', None),
                    'average': order.get('average', None),
                    'datetime': order.get('datetime', None),
                    'filled': order.get('filled', None),
                    'id': order.get('id', None),
                    'price': order.get('price', None),
                    'remaining': order.get('remaining', None),
                    'side': order.get('side', None),
                    'status': order.get('status', None),
                    'symbol': order.get('symbol', None),
                    'type': order.get('type', None),
                    'timestamp': order.get('timestamp', None),
                    'timeInForce': order.get('timeInForce', None),
                    'reduceOnly': order.get('reduceOnly', None),
                    'stopLossPrice': order.get('stopLossPrice', None),
                    'takeProfitPrice': order.get('takeProfitPrice', None)
                })

            return open_orders_list
        else:
            return None

    def get_trade_history_data(self):
        th = self._exchange.get_trade_history()
        if not th:
            return None

        th = self._exchange.process_last_trade_with_sl_tp(th)
        return th

    def get_price_action_data(self):
        oi = self.preprocess_open_interest()
        fr = self.preprocess_funding_rate()
        df = self.get_ohcl_data()

        self.last_close_price = df.iloc[-1]['C']
        self.last_candle_timestamp = df.iloc[-1]['timeframe']

        df.set_index('timeframe', inplace=True)
        df['atr_20'] = calculate_atr(df, period=20)
        df['atr_50'] = calculate_atr(df, period=50)
        df['sma_20'] = calculate_sma(df, period=20)
        df['sma_50'] = calculate_sma(df, period=50)
        df['sma_100'] = calculate_sma(df, period=100)
        df['sma_200'] = calculate_sma(df, period=200)
        df['rsi_14'] = calculate_rsi(df, period=14)
        df['macd_12_26'], df['macd_signal_9'] = calculate_macd(df)
        df['bb_middle_20'], df['bb_upper_20'], df['bb_lower_20'] = calculate_bollinger_bands(df)
        df['stochastic_k_14_s3'], df['stochastic_d_14_s3'] = calculate_stochastic_oscillator(df)
        df['adx_20'] = calculate_adx(df, n_periods=20)
        df['obv'] = round_series(calculate_obv(df), 0)
        df['obv_sma_20'] = round_series(calculate_sma(df, period=20, column='obv'), 0)

        df = shorten_large_numbers(df, 'obv')
        df = shorten_large_numbers(df, 'obv_sma_20')

        fib_dict = calculate_auto_fibonacci(df, lookback_periods=[50, 100])
        pp_dict = calculate_pivot_points(df, lookback_periods=[20, 50])

        merged_df = df.copy()
        if oi is not None:
            merged_df = pd.merge(df, oi, how='outer', left_index=True, right_index=True)
        if fr is not None:
            merged_df = pd.merge(merged_df, fr, how='outer', left_index=True, right_index=True)

        merged_df = merged_df.drop(['datetime'], axis=1)
        df_tail = merged_df.tail(self.df_tail_for_agent)
        price_data_csv = df_tail.to_csv()

        price_data_dict = {
            'price_and_indicators': price_data_csv,
            'fib_levels': fib_dict,
            'pivot_points': pp_dict
        }
        return price_data_dict

    def create_llm_input_dict(self):
        llm_input = {
            'symbol': self._exchange.market_futures,
            'price_data': self.price_action_data,
            'opened_positions': self.opened_positions,
            'opened_orders': self.opened_orders,
            'last_closed_trade': self.trade_history,
            'last_agent_output': self.last_agent_output,
            'exchange_settings': self.exchange_settings
        }

        return str(llm_input)

    def check_constrains(self):
        """ add all the 'before call conditions/constrains here """
        max_amount = self.exchange_settings.get('max_amount_based_on_free_capital', 0)
        min_amount = self.exchange_settings.get('min_amount', 0)

        if min_amount > max_amount:
            raise ConditionVerificationError(f'min amount ({min_amount}) > '
                                             f'max amount ({max_amount}, based on free capital)\n'
                                             f'exiting llm trader w/ a call')

        _logger.info("No constrains, can call the agent...")

    @staticmethod
    def generate_functions():
        # TODO: adjust this (nullable fields) and prompt for this agent to use function call
        return [
            {
                "name": "trading_decision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "amount": {"type": "number"},
                        "entry_price": {"type": "number"},
                        "order_id": {"type": "string"},
                        "order_type": {"type": "string"},
                        "rationale": {"type": "string"},
                        "stop_loss": {"type": "number"},
                        "take_profit": {"type": "number"}
                    }
                }
            }
        ]

    def call_agent(self):
        config = model_config.Gpt4Config
        _logger.info(f"creating agent using llm factory, config: {config.MODEL}")
        llm_client = LLMClientFactory.create_llm_client(config)
        functions = self.generate_functions()
        _logger.info(f"Calling the agent...")

        response = llm_client.call_with_functions(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=self.llm_input_data,
            functions=functions
        )
        parsed_output = response.choices[0].message.function_call.arguments
        structured_data = json.loads(parsed_output)
        _logger.info(f"agent call success")

        return self.agent_action_obj(**structured_data)

    def trade(self):
        self.check_constrains()
        agent_action = self.call_agent()

        order = None
        if agent_action.action == 'long' or agent_action.action == 'short':
            order = self._exchange.order(action_key=agent_action.action,
                                         amount=agent_action.amount,
                                         limit_price=agent_action.entry_price,
                                         stop_loss=agent_action.stop_loss,
                                         take_profit=agent_action.take_profit)
            msg = f"Agent entered position in strategy: {self.strategy_settings.id}:\n {asdict(agent_action)}"
            _logger.info(msg)
            _notifier.info(msg, echo='here')

        elif agent_action.action == 'close':
            order = self._exchange.order(agent_action.action, agent_action.amount)
            msg = f"Agent closed position in strategy: {self.strategy_settings.id}:\n {asdict(agent_action)}"
            _logger.info(msg)
            _notifier.info(msg, echo='here')

        elif agent_action.action == 'cancel':
            order = self._exchange.cancel_order(agent_action.order_id)
            msg = f"Agent canceled order in strategy: {self.strategy_settings.id}:\n {asdict(agent_action)}"
            _logger.info(msg)
            _notifier.info(msg, echo='here')
        else:
            msg = f"Agent holds in strategy {self.strategy_settings.id}:\n {asdict(agent_action)}"
            _logger.info(msg)
            _notifier.info(msg)

        self.save_agent_action(agent_action, order)

    def save_agent_action(self, agent_action: AgentAction, order=None):
        _logger.info("Saving agent action...")
        if order:
            order = json.dumps(order)

        with self._database.session_manager() as session:
            action_object = AgentActions(
                action=agent_action.action,
                rationale=agent_action.rationale,
                agent_output=asdict(agent_action),
                strategy_id=self.strategy_settings.id,
                order=order,
                candle_timestamp=int(self.last_candle_timestamp)
            )
            session.add(action_object)

        _logger.info("agent action saved")


class LmmTurtleValidator(LmmTrader):
    SYSTEM_PROMPT = llm_turtle_validator

    def __init__(self,
                 exchange: BaseExchangeAdapter,
                 strategy_settings: StrategySettings = None,
                 db: PostgresqlAdapter = None,
                 load_data=True,
                 ):
        super().__init__(exchange, strategy_settings, db, load_data)

    @property
    def repeated_call_time_test(self):
        repeated_call_time_test = True
        if self.last_agent_output:
            last_call = self.last_agent_output['timestamp']
            now = datetime.now(timezone.utc)
            diff = now - last_call
            repeated_call_time_test = diff > timedelta(minutes=VALIDATOR_REPEATED_CALL_TIME_TEST_MIN)
        return repeated_call_time_test

    def create_llm_input_dict(self):
        return {
            'symbol': self._exchange.market_futures,
            'price_data': self.price_action_data,
            'opened_positions': self.opened_positions,
        }

    def expand_llm_input_dict(self, last_open_position):
        if self.last_agent_output:
            self.opened_positions['stopLossPrice'] = last_open_position
        self.llm_input_data = str(self.llm_input_data)

    @staticmethod
    def generate_functions():
        return [
            {
                "name": "trading_decision",
                "description": "Provide a trading decision based on market indicators.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add_position", "wait", "set_stop_loss"]
                        },
                        "rationale": {"type": "string"},
                        "stop_loss": {"type": "number", "nullable": True}
                    },
                    "required": ["action", "rationale"]
                }
            }
        ]

    def get_last_agent_output(self):
        ao = super().get_last_agent_output()
        if not ao:
            return None

        agent_action_dict: Dict[str, str] = ao['agent_output']
        ao['agent_output'] = {k: v for k, v in agent_action_dict.items()
                              if k in ['action', 'rationale', 'stop_loss']}
        return ao


