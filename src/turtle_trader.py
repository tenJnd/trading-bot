import logging
import traceback
import uuid
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
import pandas.io.sql as sqlio
from database_tools.adapters.postgresql import PostgresqlAdapter
from retrying import retry
from slack_bot.notifications import SlackNotifier
from sqlalchemy import func, desc
from sqlalchemy.exc import OperationalError, TimeoutError

from config import (TRADE_RISK_ALLOCATION,
                    MAX_ONE_ASSET_RISK_ALLOCATION,
                    ATR_PERIOD,
                    TURTLE_ENTRY_DAYS,
                    TURTLE_EXIT_DAYS,
                    SLACK_URL,
                    VALIDATOR_REPEATED_CALL_TIME_TEST_MIN,
                    ACCEPTABLE_ROUNDING_PERCENT_THRESHOLD)
from exchange_adapter import BaseExchangeAdapter
from model.turtle_model import StrategySettings
from src.config import LOOKER_URL, MIN_POSITION_THRESHOLD
from src.llm_trader import LmmTurtlePyramidValidator, LmmTurtleEntryValidator
from src.model import trader_database
from src.model.turtle_model import Order, DepositsWithdrawals
from src.schemas.turtle_schema import OrderSchema
from src.utils.utils import save_json_to_file, get_adjusted_amount, calculate_atr, turtle_trading_signals_adjusted

_logger = logging.getLogger(__name__)
_notifier = SlackNotifier(SLACK_URL, __name__, __name__)


class AssetAllocationOverRiskLimit(Exception):
    """ Asset trades exceeds risk limit"""


def generate_trade_id():
    return str(uuid.uuid4().fields[-1])[:8]


def retry_if_sqlalchemy_transient_error(exception):
    """Determine if the exception is a transient SQLAlchemy error."""
    return isinstance(exception, (OperationalError, TimeoutError))


@dataclass
class LastOpenedPosition:
    id: str
    agg_trade_id: str
    action: str
    price: float
    cost: float
    stop_loss_price: float
    atr: float
    free_balance: float
    pl: float
    atr_period_ratio: float

    def is_long(self):
        return self.action == 'long'

    def get_atr_price_ratio(self):
        return self.atr / self.price


@dataclass
class CurrMarketConditions:
    timeframe: int
    O: float
    H: float
    L: float
    C: float
    V: float
    datetime: str
    atr_20: float
    atr_50: float
    high_20: float
    low_20: float
    high_10: float
    low_10: float
    long_entry: bool
    short_entry: bool
    long_exit: bool
    short_exit: bool

    def log_current_market_conditions(self):
        _logger.info(f'\nLONG ENTRY cond: {self.long_entry}\n'
                     f'LONG EXIT cond: {self.long_exit}\n'
                     f'SHORT ENTRY cond: {self.short_entry}\n'
                     f'SHORT EXIT cond: {self.short_exit}')

    @property
    def atr_period_ratio(self):
        return self.atr_20 / self.atr_50


class TurtleTrader:

    def __init__(self,
                 exchange: BaseExchangeAdapter,
                 strategy_settings: StrategySettings = None,
                 db: PostgresqlAdapter = None,
                 testing_file_path: bool = False,
                 ):
        self.strategy_settings = strategy_settings
        self._exchange = exchange
        self._database = trader_database if not db else db

        self.opened_positions = None
        self.last_opened_position: LastOpenedPosition = None
        self.curr_market_conditions: CurrMarketConditions = None
        self.llm_validator = None
        self.minimal_entry_cost = MIN_POSITION_THRESHOLD

        self.get_opened_positions()
        self.get_curr_market_conditions(testing_file_path)

    @property
    def n_of_opened_positions(self):
        return len(self.opened_positions) if self.opened_positions is not None else None

    @property
    def opened_positions_ids(self):
        if self.opened_positions is not None:
            return self.opened_positions['id'].to_list()
        return None

    def get_ticker_exchange_string(self, objective: str = ''):
        return (f"\n=== {str.upper(objective)} ===\n "
                f"{self._exchange.market} on {self._exchange.exchange_id}, "
                f"si: {self.strategy_settings.id}\n")

    @retry(retry_on_exception=retry_if_sqlalchemy_transient_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=2000)
    def get_opened_positions(self):
        _logger.info('Getting opened positions')
        with self._database.get_session() as session:
            df = sqlio.read_sql(
                session.query(
                    Order.id,
                    Order.agg_trade_id,
                    Order.action,
                    Order.price,
                    Order.cost,
                    Order.stop_loss_price,
                    Order.atr,
                    Order.free_balance,
                    Order.pl,
                    Order.atr_period_ratio
                ).filter(
                    Order.position_status == 'opened',
                    Order.strategy_id == self.strategy_settings.id
                ).order_by(
                    Order.timestamp
                ).statement,
                session.bind
            )

        if df.empty:
            _logger.info('No opened positions')
            self.opened_positions = None
            self.last_opened_position = None
        else:
            _logger.info(f'There are opened positions')
            self.opened_positions = df
            last_opened_position = df.iloc[-1].to_dict()
            self.last_opened_position = LastOpenedPosition(**last_opened_position)

    @retry(retry_on_exception=retry_if_sqlalchemy_transient_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=2000)
    def get_pl(self):
        _logger.info('Getting positions summary')
        with self._database.get_session() as session:
            # query last closed position
            last_closed_position_pl = session.query(
                Order.pl, Order.pl_percent
            ).filter(
                Order.action == 'close',
                Order.strategy_id == self.strategy_settings.id
            ).order_by(desc(Order.timestamp)).first()

            # Query for the sum of P&L for opened positions with the specific symbol
            strategy_pl = session.query(
                func.sum(Order.pl).label('filtered_total_pl')
            ).filter(
                Order.strategy_id == self.strategy_settings.id
            ).scalar()

            # Query for the sum of P&L for all positions
            total_pl = session.query(
                func.sum(Order.pl).label('total_pl')
            ).scalar()

            invested_capital = session.query(
                func.sum(DepositsWithdrawals.value).label('invested_capital')
            ).scalar()
            total_pl_percent = (total_pl / invested_capital) * 100

            # If there are no records matching the filters, set the values to 0.0
            strategy_pl = 0.0 if strategy_pl is None else float(strategy_pl)
            total_pl = 0.0 if total_pl is None else float(total_pl)

        return {
            'last_closed_position_pl': last_closed_position_pl,
            'strategy_pl': round(strategy_pl, 1),
            'total_pl': round(total_pl, 1),
            'total_pl_percent': round(total_pl_percent, 1)
        }

    def log_total_pl(self):
        pl: Dict[str, Any] = self.get_pl()
        msg = (
                f"{self.get_ticker_exchange_string('closed')}" +
                f"last trade P/L: {pl['last_closed_position_pl'][0]}$, "
                f"{pl['last_closed_position_pl'][1]}%\n"
                f"strategy P/L: {pl['strategy_pl']}$\n"
                f"Total P/L: {pl['total_pl']}$, {pl['total_pl_percent']}%\n"
                f"{LOOKER_URL}"
        )
        _logger.info(msg)
        _notifier.info(msg, echo='here')

    def calculate_pl(self, close_order: OrderSchema):
        if self.last_opened_position.is_long():
            total_cost = float(self.opened_positions.cost.sum())
            total_revenue = float(close_order.cost)
        else:
            total_cost = float(close_order.cost)
            total_revenue = float(self.opened_positions.cost.sum())

        pl = total_revenue - total_cost
        pl_percent = (pl / total_cost) * 100

        return round(float(pl), 2), round(float(pl_percent), 2)

    def get_curr_market_conditions(self, testing_file_path: str = None):
        if testing_file_path:
            ohlc = pd.read_csv(testing_file_path)
        else:
            ohlc = self._exchange.fetch_ohlc(days=self.strategy_settings.buffer_days,
                                             timeframe=self.strategy_settings.timeframe)

        ohlc['atr_20'] = calculate_atr(ohlc, period=ATR_PERIOD)
        ohlc['atr_50'] = calculate_atr(ohlc, period=50)
        ohlc = turtle_trading_signals_adjusted(ohlc)

        curr_market_con = ohlc.iloc[-1].to_dict()
        self.curr_market_conditions = CurrMarketConditions(**curr_market_con)
        self.curr_market_conditions.log_current_market_conditions()
        return ohlc

    def create_agg_trade_id(self):
        if self.last_opened_position is None:
            return generate_trade_id()
        else:
            return self.last_opened_position.agg_trade_id

    def get_stop_loss_price(self, action, atr):
        atr2 = self.strategy_settings.stop_loss_atr_multipl * atr * self.curr_market_conditions.atr_period_ratio
        if action == 'long':  # long
            return self.curr_market_conditions.C - atr2
        elif action == 'short':  # short
            return self.curr_market_conditions.C + atr2
        else:
            return

    def get_atr_for_pyramid(self):
        # # lower atr for entry to half for pyramid trade
        # # if atr/price ration is lower than 2% (less volatile market)
        # atr_ratio = self.last_opened_position.get_atr_price_ratio()
        # if atr_ratio < self.strategy_settings.aggressive_price_atr_ratio:
        #     return (self.last_opened_position.atr *
        #             self.strategy_settings.aggressive_pyramid_entry_multipl *
        #             self.curr_market_conditions.atr_period_ratio)
        pyramid_atr = (
                self.last_opened_position.atr *
                self.strategy_settings.pyramid_entry_atr_multipl *
                self.last_opened_position.atr_period_ratio  # can be current or last atr ratio
        )

        long_pyramid_price = self.last_opened_position.price + pyramid_atr
        short_pyramid_price = self.last_opened_position.price - pyramid_atr
        return long_pyramid_price, short_pyramid_price

    def recalc_limited_free_entry_balance(self, free_balance, total_balance):
        if self.last_opened_position:
            # This is to ensure that the pyramid position is not larger than the last position
            # (checking the free balance at the time of the last position).
            # In the case of trading several assets.
            last_pos_free_balance = self.last_opened_position.free_balance
            if free_balance > last_pos_free_balance:
                _logger.info('Balance is greater than last position free balance '
                             '-> setting balance to last open position free balance')
                free_balance = last_pos_free_balance

            actual_asset_allocation = self.opened_positions.cost.sum() / total_balance
            if actual_asset_allocation > MAX_ONE_ASSET_RISK_ALLOCATION:
                _logger.warning(f'This trade would excess max capital allocation into one asset'
                                f'actual current allocation: {actual_asset_allocation},'
                                f'max one asset risk allocation: {MAX_ONE_ASSET_RISK_ALLOCATION}')
                raise AssetAllocationOverRiskLimit

            return free_balance
        else:
            return free_balance

    @retry(retry_on_exception=retry_if_sqlalchemy_transient_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=2000)
    def update_closed_orders(self):
        _logger.info('Updating closed orders in db')
        with self._database.session_manager() as session:
            session.query(Order).filter(Order.id.in_(self.opened_positions_ids)).update(
                {"position_status": "closed"},
                synchronize_session=False  # Use 'fetch' if objects are being used in the session
            )
        _logger.info('Closed orders successfully updated')

    @retry(retry_on_exception=retry_if_sqlalchemy_transient_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=2000)
    def commit_order_to_db(self, order_object: OrderSchema):
        with self._database.session_manager() as session:
            session.add(order_object)
        _logger.info('Order successfully saved')

    def update_stop_loss(self, stop_loss_price):
        # TODO check agent stop loss
        _logger.info(f"Updating stop loss to price: {stop_loss_price}...")
        with self._database.session_manager() as session:
            session.query(Order).filter(
                Order.id == self.last_opened_position.id
            ).update(
                {"stop_loss_price": float(stop_loss_price)},
                synchronize_session=False  # Use 'fetch' if objects are being used in the session
            )
        _logger.info("Stop-loss successfully updated")

    def save_order(self, order, action, position_status='opened'):
        _logger.info('Saving order to file and DB')

        if order.get('status', None) is None:
            try:
                order_id = order.get('id')
                order = self._exchange.fetch_order(order_id)
            except Exception as exc:
                _logger.warning(f"Could not fetch the order, saved order might be incomplete: {exc}")
                _logger.warning(f"Could not fetch the order, saved order might be incomplete: {exc}")

        self._exchange.fetch_balance()

        order_object: OrderSchema = OrderSchema().load(order)
        order_object.atr = self.curr_market_conditions.atr_20
        order_object.atr_period_ratio = self.curr_market_conditions.atr_period_ratio
        order_object.action = action
        order_object.free_balance = self._exchange.free_balance
        order_object.total_balance = self._exchange.total_balance
        order_object.position_status = position_status
        order_object.agg_trade_id = self.create_agg_trade_id()

        order_object.stop_loss_price = self.get_stop_loss_price(action, order_object.atr)

        order_object.exchange = self._exchange.exchange_id
        order_object.contract_size = self._exchange.contract_size
        order_object.strategy_id = self.strategy_settings.id
        order_object.candle_timeframe = self.curr_market_conditions.timeframe

        if action == 'close':
            order_object.closed_positions = self.opened_positions_ids
            order_object.pl, order_object.pl_percent = self.calculate_pl(order_object)

        try:
            save_json_to_file(order_object, f"order_{order_object.id}")
        except Exception as exc:
            _logger.error(f"Cannot save json file, skipp. {exc}")
            _notifier.error(f"Cannot save json file, skipp. {exc}")

        self.commit_order_to_db(order_object)

    def calculate_entry_amount(self):
        self._exchange.fetch_balance()
        free_balance = self._exchange.free_balance
        total_balance = self._exchange.total_balance
        free_balance = self.recalc_limited_free_entry_balance(free_balance, total_balance)

        trade_risk_cap = free_balance * TRADE_RISK_ALLOCATION
        raw_amount = (
                             trade_risk_cap /
                             (
                                     self.strategy_settings.stop_loss_atr_multipl *
                                     self.curr_market_conditions.atr_20 *
                                     self.curr_market_conditions.atr_period_ratio
                             )
                     ) / self._exchange.contract_size

        _logger.debug(f"Raw calculated amount: {raw_amount}")

        # Check if the raw amount is below the minimum tradable amount
        if raw_amount < self._exchange.min_amount:
            msg = (self.get_ticker_exchange_string('raw amount below minimum') +
                   f"Calculated raw amount {raw_amount} is below the minimum tradable amount "
                   f"{self._exchange.min_amount}. SKIPPING")
            _logger.warning(msg)
            _notifier.warning(msg)
            return None

        # Calculate cost before rounding
        raw_cost = raw_amount * self.curr_market_conditions.C * self._exchange.contract_size
        _logger.debug(f"Cost before rounding: {raw_cost}")

        # Apply exchange precision rounding
        rounded_amount = get_adjusted_amount(raw_amount, self._exchange.amount_precision)
        _logger.debug(f"Rounded amount: {rounded_amount}")

        # Calculate cost after rounding
        rounded_cost = rounded_amount * self.curr_market_conditions.C * self._exchange.contract_size
        _logger.debug(f"Cost after rounding: {rounded_cost}")

        # Ensure costs are within acceptable percentage difference
        percentage_difference = abs(rounded_cost - raw_cost) / raw_cost * 100
        if percentage_difference > ACCEPTABLE_ROUNDING_PERCENT_THRESHOLD:
            msg = (self.get_ticker_exchange_string('cost mismatch') +
                   f"Percentage difference between pre-rounded cost ({raw_cost}) "
                   f"and post-rounded cost ({rounded_cost}) is {percentage_difference:.2f}%, "
                   f"exceeding the threshold of {ACCEPTABLE_ROUNDING_PERCENT_THRESHOLD}%. SKIPPING")
            _logger.warning(msg)
            _notifier.warning(msg)
            return None

        # check if cost is above minimal entry cost constant
        minimal_entry_cond = rounded_cost < self.minimal_entry_cost
        if minimal_entry_cond:
            msg = (self.get_ticker_exchange_string('min entry condition') +
                   f"Cost {rounded_cost} is lower than "
                   f"free_balance {free_balance}, "
                   f"or lower than minimal entry cost constraint {self.minimal_entry_cost}. SKIPPING")
            _logger.warning(msg)
            _notifier.warning(msg)
            return None

        return rounded_amount

    def entry_position(self, action):
        amount = self.calculate_entry_amount()
        if not amount:
            return

        _logger.info(f"Amount: {amount}, Contract size: {self._exchange.contract_size}")
        order = self._exchange.order(action, amount)

        if order:
            self.save_order(order, action)

        msg = (self.get_ticker_exchange_string('Entered') +
               f"Position: {action}\n"
               f"Amount: {amount}\n"
               f"Contract_size: {self._exchange.contract_size}\n"
               )
        _logger.info(msg)
        _notifier.info(msg, echo='here')

    def exit_position(self):
        action = 'close'
        order = self._exchange.order(action)
        if order:
            self.save_order(order, action, position_status='closed')
            self.update_closed_orders()
            self.log_total_pl()

    def create_llm_validator(self) -> LmmTurtlePyramidValidator:
        validator = self.llm_validator(self._exchange, self.strategy_settings, self._database)
        if self.last_opened_position:
            validator.expand_llm_input_dict(self.last_opened_position.stop_loss_price)
        else:
            validator.expand_llm_input_dict()
        return validator

    def process_agent_action(self, agent_action, side):
        msg = ""
        if agent_action.action == 'add_position':
            msg = (self.get_ticker_exchange_string('lmm add position') +
                   f"Lmm validator is adding to position\n"
                   f"rationale: {agent_action.rationale}")
            self.entry_position(side)
        elif agent_action.action == 'enter_position':
            msg = (self.get_ticker_exchange_string('lmm enter position') +
                   f"Lmm validator is entering the position\n"
                   f"rationale: {agent_action.rationale}")
            self.entry_position(side)
        elif agent_action.action == 'set_stop_loss':
            stop_loss = agent_action.stop_loss
            if stop_loss:
                msg = (self.get_ticker_exchange_string('llm stop-loss') +
                       "Llm validator is setting new stop-loss\n"
                       f"curr price: {self.curr_market_conditions.C}, "
                       f"atr_20: {self.curr_market_conditions.atr_20}\n"
                       f"stop-loss: {agent_action.stop_loss}\n"
                       f"rationale: {agent_action.rationale}")
                self.update_stop_loss(stop_loss)
        else:
            msg = (self.get_ticker_exchange_string('llm hold') +
                   "LMM validator action is to wait. "
                   "We will wait for another run\n"
                   f"action: {agent_action.action}, rationale: {agent_action.rationale}")

        _logger.info(msg)
        _notifier.info(msg, echo='here')

    def entry_w_validation(self, side):
        validator = self.create_llm_validator()

        # check if validator have run in VALIDATOR_REPEATED_CALL_TIME_TEST_MIN
        # and skipp validation if test not passed (False)
        if not validator.repeated_call_time_test:
            msg = (self.get_ticker_exchange_string('llm repeated call') +
                   "LMM validator was called in less then "
                   f"{VALIDATOR_REPEATED_CALL_TIME_TEST_MIN} minutes.\n"
                   "We will wait for another run..")
            _logger.info(msg)
            _notifier.info(msg)
        else:
            try:
                agent_action = validator.call_agent()
                self.process_agent_action(agent_action, side)
                validator.save_agent_action(agent_action)
            except Exception as exc:
                msg = (self.get_ticker_exchange_string('lmm call exception') +
                       f"{str(exc)}, traceback: {traceback.format_exc()}"
                       f"There was a problem with agent call - SKIPPING ticker")
                _logger.error(msg)
                _notifier.error(msg, echo='here')

    def process_opened_position(self):
        _logger.info('Processing opened positions')
        self.llm_validator = LmmTurtlePyramidValidator

        curr_mar_cond = self.curr_market_conditions
        last_stop_loss = self.last_opened_position.stop_loss_price
        long_pyramid_price, short_pyramid_price = self.get_atr_for_pyramid()

        # check if number of pyramid trade is over limit
        pyramid_stop = self.n_of_opened_positions > self.strategy_settings.pyramid_entry_limit
        self.minimal_entry_cost = self.minimal_entry_cost / 2
        _logger.info(f"Setting minimal entry cost to {self.minimal_entry_cost}")

        if self.last_opened_position.is_long():
            # exit position
            if curr_mar_cond.long_exit:
                _logger.info('Exiting long position/s')
                self.exit_position()
            # exit position -> stop loss
            elif curr_mar_cond.C <= last_stop_loss:
                _logger.info('Initiating long stop-loss')
                self.exit_position()
            # add to position -> pyramiding
            elif curr_mar_cond.C >= long_pyramid_price and not pyramid_stop:
                _logger.info(f'Adding to long position -> check w LLM validator')
                self.entry_w_validation('long')
            else:
                _logger.info('Staying in position '
                             '-> no condition for opened position is met')
        else:
            # exit position
            if curr_mar_cond.short_exit:
                _logger.info('Exiting short position/s')
                self.exit_position()
            # exit position -> stop loss
            elif curr_mar_cond.C >= last_stop_loss:
                _logger.info('Initiating short stop-loss')
                self.exit_position()
            # add to position -> pyramiding
            elif curr_mar_cond.C <= short_pyramid_price and not pyramid_stop:
                _logger.info(f'Adding to short position -> check w LLM validator')
                self.entry_w_validation('short')
            else:
                _logger.info('Staying in position '
                             '-> no condition for opened position is met')

    def trade(self):
        if self.opened_positions is None:
            self.llm_validator = LmmTurtleEntryValidator
            curr_cond = self.curr_market_conditions
            # entry long
            if curr_cond.long_entry and not curr_cond.long_exit:  # safety
                _logger.info('Long cond is met -> entering long position')
                self.entry_w_validation('long')
            # entry short
            elif curr_cond.short_entry and not curr_cond.short_exit:  # safety
                _logger.info('Short cond is met -> entering short position')
                self.entry_w_validation('short')
            # do nothing
            else:
                _logger.info('No opened positions and no condition is met for entry -> SKIPPING')
        # work with opened position
        else:
            self.process_opened_position()

    def close_position(self):
        if self.opened_positions is not None:
            self.exit_position()
