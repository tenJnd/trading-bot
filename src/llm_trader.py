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
from src.config import LLM_TRADER_SLACK_URL, VALIDATOR_REPEATED_CALL_TIME_TEST_MIN, \
    ACCEPTABLE_ROUNDING_PERCENT_THRESHOLD
from src.model import trader_database
from src.model.turtle_model import AgentActions
from src.prompts import llm_trader_prompt, turtle_pyramid_validator_prompt, turtle_entry_validator_prompt
from src.utils.utils import (calculate_sma, round_series,
                             shorten_large_numbers,
                             dynamic_safe_round, calculate_indicators_for_llm_trader,
                             calculate_indicators_for_llm_entry_validator, StrategySettingsModel, get_adjusted_amount,
                             calculate_closest_fvg_zones, calculate_fib_levels_pivots)

_logger = logging.getLogger(__name__)
_notifier = SlackNotifier(url=LLM_TRADER_SLACK_URL, username='main')

# Example dictionary
PERIODS = {
    '4h': 40,
    '1d': 220,
    '3d': 660,
}


def get_next_key(base_key):
    # Convert the keys to a list
    keys = list(PERIODS.keys())

    # Find the index of the key
    index = keys.index(base_key)

    # Get the next key and value if it exists
    if index + 1 < len(keys):
        next_key = keys[index + 1]
    else:
        raise ValueError(f"No key exists after '{base_key}'")

    return next_key


class ValidationError(Exception):
    """ Validation error"""


class TraderModel(model_config.ModelConfig):
    MODEL = 'gpt-4o'
    MAX_TOKENS = 2000
    CONTEXT_WINDOW = 8192
    TEMPERATURE = 0.4  # Keep outputs deterministic for scoring and ranking
    RESPONSE_TOKENS = 500  # Ensure response fits within limits
    FREQUENCY_PENALTY = 0.0  # Avoid repetition in rationale
    PRESENCE_PENALTY = 0.3  # Encourage new ideas or highlighting unique patterns


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
    data: dict = None

    def nice_print(self):
        d = asdict(self)
        return '\n'.join([f"{k}: {v}" for k, v in d.items() if v is not None])

    @property
    def is_long(self):
        return self.action == 'long'

    @property
    def is_short(self):
        return self.action == 'short'

    @property
    def is_entry(self):
        return self.action in ['long', 'short']

    @property
    def is_close(self):
        return self.action == 'close'

    @property
    def is_cancel(self):
        return self.action == 'cancel'

    @property
    def is_update_sl(self):
        return self.action == 'update_sl'

    @property
    def is_update_tp(self):
        return self.action == 'update_tp'

    @property
    def is_hold(self):
        return self.action == 'hold'

    def rr_ratio(self, close_price):
        take_profit = self.take_profit
        if self.is_long:
            take_profit = take_profit or float('inf')
        if self.is_short:
            take_profit = take_profit or 0

        move_against = abs(self.stop_loss - close_price)
        move_in_favour = abs(take_profit - close_price)
        return move_in_favour / move_against


class LlmTrader:
    agent_name = 'llm_trader'
    system_prompt = llm_trader_prompt
    agent_action_obj = AgentAction
    llm_model_config = TraderModel
    df_tail_for_agent = 20
    leverage = 2

    def __init__(self,
                 exchange: BaseExchangeAdapter,
                 strategy_settings: StrategySettingsModel = None,
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
            self.trade_history = self.get_last_trade_data()
            self.last_agent_output = self.get_last_agent_output()
            self.exchange_settings = self.get_exchange_settings()

            self.llm_input_data = self.create_llm_input_dict()

    @classmethod
    def init_just_exchange(cls,
                           exchange: BaseExchangeAdapter,
                           strategy_settings: StrategySettingsModel = None,
                           db: PostgresqlAdapter = None,
                           ):
        return cls(exchange=exchange, strategy_settings=strategy_settings, db=db, load_data=False)

    def get_timing_info(self, timeframe, last_candle_timestamp):
        if not timeframe:
            timeframe = self.strategy_settings.timeframe

        return {
            'candle_timeframe': timeframe,
            'candle_timestamp': last_candle_timestamp,
            'current_timestamp': int(datetime.now().timestamp()),
        }

    def get_last_agent_output(self):
        if self.opened_orders or self.opened_positions:
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
        else:
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
        max_amount = dynamic_safe_round(free_balance / self.last_close_price, 2)
        return {
            'min_cost': self._exchange.min_cost,
            'min_amount': self._exchange.min_amount,
            'free_capital': free_balance,
            'total_capital': total_balance,
            'max_amount_based_on_free_capital': max_amount
        }

    def preprocess_open_interest(self, timeframe=None):
        timeframe = timeframe if timeframe else self.strategy_settings.timeframe
        oi = self._exchange.get_open_interest_hist(timeframe=timeframe)
        if oi:
            oi_df = pd.DataFrame(oi)
            keep_cols = ['timestamp', 'open_interest']
            oi_df['open_interest'] = round_series(oi_df['openInterestValue'], 0)
            oi_df = oi_df[keep_cols]
            oi_df['open_interest_sma_20'] = calculate_sma(oi_df, 20, column='open_interest')
            oi_df['open_interest_sma_10'] = calculate_sma(oi_df, 10, column='open_interest')
            oi_df.set_index('timestamp', inplace=True)
            return oi_df
        else:
            return None

    def get_current_funding_rate(self):
        fr = self._exchange.get_funding_rate()
        simple_fr_dict = {}
        if fr:
            simple_fr_dict['funding_rate'] = fr.get('fundingRate', 0)
            simple_fr_dict['funding_timestamp'] = fr.get('fundingTimestamp', 0)
            simple_fr_dict['previous_funding_rate'] = fr.get('previousFundingRate', None)
            simple_fr_dict['previous_funding_timestamp'] = fr.get('previousFundingTimestamp', None)
            return simple_fr_dict

    def get_ohcl_data(self, buffer_days=None, timeframe=None):
        timeframe = timeframe if timeframe else self.strategy_settings.timeframe
        buffer_days = buffer_days if buffer_days else self.strategy_settings.buffer_days
        ohlc = self._exchange.fetch_ohlc(days=buffer_days, timeframe=timeframe)
        return ohlc

    def get_open_positions_data(self):
        op = self._exchange.get_opened_positions()[0]
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
                    'order_type': order['info'].get('stopOrderType', None),
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
                    'stopLossPrice': order.get('stopLossPrice', None),
                    'takeProfitPrice': order.get('takeProfitPrice', None)
                })

            return open_orders_list
        else:
            return None

    def get_last_trade_data(self):
        th = self._exchange.get_trade_history()
        if not th:
            return None

        th = self._exchange.process_last_trade_with_sl_tp(th)
        return th

    @staticmethod
    def _calculate_indicators_for_llm_trader(df):
        return calculate_indicators_for_llm_trader(df)

    def calculate_data_multiple_periods(self):
        base_timeframe = self.strategy_settings.timeframe
        higher_timeframe = get_next_key(base_timeframe)

        timeframes = [base_timeframe, higher_timeframe]

        result_data = {}

        for timeframe in timeframes:
            buffer_days = PERIODS.get(timeframe)
            oi = self.preprocess_open_interest(timeframe=timeframe)
            df = self.get_ohcl_data(buffer_days=buffer_days, timeframe=timeframe)

            self.last_close_price = df.iloc[-1]['C']
            last_candle_timestamp = df.iloc[-1]['timeframe']

            if timeframe == self.strategy_settings.timeframe:
                self.last_candle_timestamp = df.iloc[-1]['timeframe']

            df.set_index('timeframe', inplace=True)
            df = self._calculate_indicators_for_llm_trader(df)

            df = shorten_large_numbers(df, 'obv')
            df = shorten_large_numbers(df, 'obv_sma_20')

            if timeframe == '1d':
                fib_depth = 10
                tail = int(self.df_tail_for_agent / 2)
            else:
                tail = self.df_tail_for_agent
                fib_depth = 20

            fib_dict = calculate_fib_levels_pivots(df, depth=fib_depth)
            # pp_dict = calculate_pivot_points(df, lookback_periods=[20])
            fvg_dict = calculate_closest_fvg_zones(df, self.last_close_price)
            # lin_reg = calculate_regression_channels_with_slope(df, periods=[20])

            merged_df = df.copy()
            if oi is not None:
                merged_df = pd.merge(df, oi, how='outer', left_index=True, right_index=True)

            merged_df = merged_df.drop(['datetime'], axis=1)
            df_tail = merged_df.tail(tail)
            price_data_csv = df_tail.to_csv()

            timing_data = self.get_timing_info(timeframe, last_candle_timestamp)

            result_data[timeframe] = {
                'timing_info': timing_data,
                'price_and_indicators': price_data_csv,
                'fib_levels': fib_dict,
                'closest_fair_value_gaps_levels': fvg_dict,
                # 'linear_regression_channels': lin_reg
            }
        return result_data

    def get_price_action_data(self):
        fr = self.get_current_funding_rate()
        data_multiple_periods = self.calculate_data_multiple_periods()
        price_data_dict = {
            'current_price': self.last_close_price,
            'price_data': data_multiple_periods,
            'current_funding_rate': fr
        }
        return price_data_dict

    def create_llm_input_dict(self):
        llm_input = {
            'symbol': self._exchange.market_futures,
            'price_data': self.price_action_data,
            'opened_positions': self.opened_positions,
            'opened_orders': self.opened_orders,
            # 'last_closed_trade': self.trade_history,
            'last_agent_output': self.last_agent_output,
            # 'exchange_settings': self.exchange_settings
        }

        return llm_input

    @staticmethod
    def generate_functions():
        return [
            {
                "name": "trading_decision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": ["long", "short", "close", "cancel", "update_sl", "update_tp", "hold"],
                                        "description": "The trading action to perform."
                                    },
                                    "order_type": {
                                        "type": ["string", "null"],
                                        "enum": ["limit", "market"],
                                        "description": "The type of order to place. Required for 'long' or 'short'."
                                    },
                                    "amount": {
                                        "type": ["number", "null"],
                                        "description": "The amount for the position or order."
                                    },
                                    "entry_price": {
                                        "type": ["number", "null"],
                                        "description": "The price at which to enter the trade "
                                                       "(only required if order_type is 'limit')."
                                    },
                                    "stop_loss": {
                                        "type": ["number", "null"],
                                        "description": "The stop-loss price level or updated stop-loss level for 'update_sl'."
                                    },
                                    "take_profit": {
                                        "type": ["number", "null"],
                                        "description": "The take-profit price level or updated take-profit level for 'update_tp'."
                                    },
                                    "order_id": {
                                        "type": ["string", "null"],
                                        "description": "The ID of the order to cancel or update. Required for 'cancel' and 'update_sl', update_tp."
                                    },
                                    "rationale": {
                                        "type": "string",
                                        "description": "Explanation of the decision based on price action, trend, "
                                                       "indicators, open interest, funding rate, and exchange settings."
                                    }
                                },
                                "required": ["action", "rationale"],
                                "description": "Defines a single trading action."
                            }
                        }
                    },
                    "required": ["actions"],
                    "description": "Defines a list of trading actions to be performed in sequence."
                }
            }
        ]

    def call_agent(self, list_of_actions=False):
        _logger.info(f"creating agent using llm factory, config: {self.llm_model_config.MODEL}")
        llm_client = LLMClientFactory.create_llm_client(self.llm_model_config)
        functions = self.generate_functions()
        _logger.info(f"Calling the agent...")

        response = llm_client.call_with_functions(
            system_prompt=self.system_prompt,
            user_prompt=str(self.llm_input_data),
            functions=functions
        )
        _logger.info(f'response: {response}')
        parsed_output = response.choices[0].message.function_call.arguments
        structured_data = json.loads(parsed_output)
        _logger.info(f"agent call success")

        if list_of_actions:
            actions = structured_data['actions']
            actions = [self.agent_action_obj(**data) for data in actions]
            return actions
        else:
            return self.agent_action_obj(**structured_data)

    def call_agent_w_validation(self):
        counter = 1
        tries = 2
        agent_actions = None

        while counter <= tries:
            try:
                agent_actions = self.call_agent(list_of_actions=True)
                for agent_action in agent_actions:
                    self.check_trade_valid(agent_action)
                break
            except ValidationError as exc:
                msg = (f"agent output not validated: {exc} "
                       f"agent_actions: {[agent_action.nice_print() for agent_action in agent_actions]}\n"
                       f"try num: {counter}")
                _logger.warning(msg)
                _notifier.warning(msg)

                previous_output_dict = {
                    'previous_agent_action': agent_actions,
                    'error': str(exc)
                }
                self.llm_input_data['previous_output'] = previous_output_dict
                counter += 1

        # If the loop exits without breaking (all attempts failed), raise an error
        if counter > tries:
            raise ValidationError(f"Validation failed after {tries} attempts. Last agent action: "
                                  f"{getattr(agent_actions, 'nice_print', lambda: '<unknown>')()}")

        return agent_actions

    def pre_check_constrains(self):
        """ add all the 'before call conditions/constrains here """
        max_amount = self.exchange_settings.get('max_amount_based_on_free_capital', 0)
        min_amount = self.exchange_settings.get('min_amount', 0)

        if self.opened_orders or self.opened_positions:
            _logger.info("No constrains, can call the agent...")
            return

        if min_amount > max_amount:
            raise ConditionVerificationError(f'min amount ({min_amount}) > '
                                             f'max amount ({max_amount}, based on free capital)\n'
                                             f'exiting llm trader w/ a call')

        min_threshold = 150
        if self._exchange.free_balance < min_threshold:
            raise ConditionVerificationError(f'free balance is lower then '
                                             f'min threshold: {min_threshold}')

        _logger.info("No constrains, can call the agent...")

    def check_trade_valid(self, agent_action) -> AgentAction:
        """
        Validate the agent's action post-call to ensure logical and valid trade decisions.

        Args:
            agent_action (AgentAction): The action object returned by the agent.

        Returns:
            AgentAction: The validated agent action.

        Raises:
            ValidationError: If R:R ratio is invalid or prices do not align with the trade direction.
        """
        if not agent_action.is_entry:
            return agent_action

        last_close = self._exchange.get_close_price()
        c_price = agent_action.entry_price if agent_action.entry_price else last_close

        # Initialize error messages list
        error_messages = []

        min_rr_ratio = 1.5
        rr_ratio = agent_action.rr_ratio(c_price)
        if rr_ratio < min_rr_ratio:
            error_messages.append(
                f"Invalid R:R ratio. R:R is less than {min_rr_ratio} for {agent_action.action} action. "
                f"Current R:R = {rr_ratio:.2f}. "
                f"Adjust the {agent_action.action} position to ensure R:R >= {min_rr_ratio}.\n or hold."
            )

        # Price validation for long positions
        if agent_action.is_long:
            if not (agent_action.stop_loss < c_price < (agent_action.take_profit or float('inf'))):
                error_messages.append(
                    f"Invalid price logic for long position: stop-loss ({agent_action.stop_loss}) "
                    f"must be below entry price ({c_price}), "
                    f"and take-profit ({agent_action.take_profit or 'None'}) must be above entry price. "
                    f"Adjust stop-loss or take-profit levels to correct this logic.\n"
                )
            if agent_action.entry_price:
                if agent_action.entry_price > last_close:
                    error_messages.append(
                        f"Invalid limit order price for long position: limit price ({agent_action.entry_price}) "
                        f"is above the current close price ({last_close}). "
                        f"Adjust the limit price to be below the close price or use market order.\n"
                    )

        # Price validation for short positions
        if agent_action.is_short:
            if not (agent_action.stop_loss > c_price > (agent_action.take_profit or 0)):
                error_messages.append(
                    f"Invalid price logic for short position: stop-loss ({agent_action.stop_loss}) "
                    f"must be above entry price ({c_price}), "
                    f"and take-profit ({agent_action.take_profit or 'None'}) must be below entry price. "
                    f"Adjust stop-loss or take-profit levels to correct this logic.\n"
                )
            if agent_action.entry_price:
                if agent_action.entry_price < last_close:
                    error_messages.append(
                        f"Invalid limit order price for short position: limit price ({agent_action.entry_price}) "
                        f"is below the current close price ({last_close}). "
                        f"Adjust the limit price to be above the close price. or use market order.\n"
                    )

        # Raise ValidationError with all issues if any exist
        if error_messages:
            raise ValidationError("; ".join(error_messages))
        else:
            _logger.info(f"Trade is valid")

        return agent_action

    def calculate_amount(self, agent_action) -> AgentAction:
        """
        Calculate the trade amount based on the agent's action and ensure it adheres to exchange rules.

        Args:
            agent_action (AgentAction): The agent's action containing trade parameters.

        Returns:
            AgentAction: Updated with the calculated amount, or None if the trade is invalid.
        """
        # Determine the price for calculation (use limit price if available, otherwise last close price)
        self._exchange.fetch_balance()
        agent_limit_price = agent_action.entry_price
        close_price = self._exchange.get_close_price()
        c_price = agent_limit_price if agent_limit_price else close_price

        # Calculate the potential risk per trade
        move_against = abs(agent_action.stop_loss - c_price)
        trade_risk_cap = (self._exchange.free_balance * 0.9) * 0.01  # 1% risk per trade * leverage!!
        raw_amount = (trade_risk_cap / move_against) * self.leverage

        # Check if the raw amount is below the exchange's minimum tradable amount
        if raw_amount < self._exchange.min_amount:
            msg = (self.format_log(agent_action) +
                   f"Calculated raw amount {raw_amount} is below the minimum tradable amount "
                   f"{self._exchange.min_amount}. SKIPPING")
            _logger.warning(msg)
            _notifier.warning(msg)
            return None

        # Calculate raw cost based on the raw amount
        raw_cost = raw_amount * c_price * self._exchange.contract_size

        # Apply precision rounding based on exchange rules
        rounded_amount = get_adjusted_amount(raw_amount, self._exchange.amount_precision)
        _logger.debug(f"Rounded amount: {rounded_amount}")

        # Calculate cost after rounding
        rounded_cost = rounded_amount * c_price * self._exchange.contract_size
        _logger.debug(f"Cost after rounding: {rounded_cost}")

        # Validate rounding difference
        percentage_difference = abs(rounded_cost - raw_cost) / raw_cost * 100
        if percentage_difference > ACCEPTABLE_ROUNDING_PERCENT_THRESHOLD:
            msg = (f"Percentage difference between pre-rounded cost ({raw_cost}) "
                   f"and post-rounded cost ({rounded_cost}) is {percentage_difference:.2f}%, "
                   f"exceeding the threshold of {ACCEPTABLE_ROUNDING_PERCENT_THRESHOLD}%. SKIPPING")
            _logger.warning(msg)
            _notifier.warning(msg)
            return None

        # Assign the rounded amount to the agent's action and return
        agent_action.amount = rounded_amount
        return agent_action

    def format_log(self, agent_action):
        # Combine the strategy settings with the action string, skipping None values in the log
        return (f"ticker: {self.strategy_settings.ticker}, "
                f"strategy_id: {self.strategy_settings.id}\n"
                f"{agent_action.nice_print()}")

    def process_non_hold_action(self, agent_action):
        order = None
        if agent_action.is_entry:
            agent_action = self.calculate_amount(agent_action)
            if agent_action is None:
                return

            order = self._exchange.order(action_key=agent_action.action,
                                         amount=agent_action.amount,
                                         limit_price=agent_action.entry_price,
                                         stop_loss=agent_action.stop_loss,
                                         take_profit=agent_action.take_profit,
                                         leverage=self.leverage)
        elif agent_action.is_close:
            order = self._exchange.order(agent_action.action, agent_action.amount)
        elif agent_action.is_cancel:
            order = self._exchange.cancel_order(
                order_id=agent_action.order_id,
                symbol=self._exchange.market_futures
            )
        elif agent_action.is_update_sl:
            order = self._exchange.edit_order(
                ordr_id=agent_action.order_id,
                stop_loss=agent_action.stop_loss)
        elif agent_action.is_update_tp:
            order = self._exchange.edit_order(
                ordr_id=agent_action.order_id,
                take_profit=agent_action.take_profit)

        self.save_agent_action(agent_action, order)
        _notifier.info(f':bangbang: {self.format_log(agent_action)}', echo='here')

    def trade(self):
        try:
            self.pre_check_constrains()
            agent_actions = self.call_agent_w_validation()

        except ConditionVerificationError as e:
            msg = f"Conditional error; SKIPPING: {str(e)}"
            _logger.error(msg)
            _logger.warning(msg)
            return
        except ValidationError:
            return

        for agent_action in agent_actions:
            if agent_action.is_hold:
                _notifier.info(self.format_log(agent_action))
            else:
                self.process_non_hold_action(agent_action)

            _logger.info(self.format_log(agent_action))

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
                candle_timestamp=int(self.last_candle_timestamp),
                agent_name=self.agent_name
            )
            session.add(action_object)

        _logger.info("agent action saved")


class LlmTurtlePyramidValidator(LlmTrader):
    agent_name = 'lmm_turtle_pyramid_validator'
    system_prompt = turtle_pyramid_validator_prompt
    df_tail_for_agent = 10

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
            'opened_positions': self.opened_positions
        }

    def expand_llm_input_dict(self, last_open_position_stop_loss=None):
        if self.last_agent_output and self.opened_positions:
            self.opened_positions['stopLossPrice'] = last_open_position_stop_loss
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
                            "enum": ["add_position", "hold", "set_stop_loss"]
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


class LlmTurtleEntryValidator(LlmTurtlePyramidValidator):
    agent_name = 'lmm_turtle_entry_validator'
    system_prompt = turtle_entry_validator_prompt
    df_tail_for_agent = 10

    def __init__(self,
                 exchange: BaseExchangeAdapter,
                 strategy_settings: StrategySettings = None,
                 db: PostgresqlAdapter = None,
                 load_data=True,
                 ):
        super().__init__(exchange, strategy_settings, db, load_data)

    @staticmethod
    def _calculate_indicators_for_llm_trader(df):
        return calculate_indicators_for_llm_entry_validator(df)

    def create_llm_input_dict(self):
        return {
            'symbol': self._exchange.market_futures,
            'price_data': self.price_action_data
        }

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
                            "enum": ["enter_position", "hold"]
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["action", "rationale"]
                }
            }
        ]
