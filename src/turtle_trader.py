import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

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
                    SLACK_URL)
from exchange_adapter import BaseExchangeAdapter
from model.turtle_model import StrategySettings
from src.model import trader_database
from src.model.turtle_model import Order
from src.schemas.turtle_schema import OrderSchema
from src.utils.utils import save_json_to_file, get_adjusted_amount, calculate_atr

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


def turtle_trading_signals_adjusted(df):
    """
    Identify Turtle Trading entry and exit signals for both long and short positions, adjusting for early rows.

    Parameters:
    - df: pandas DataFrame with at least 'High' and 'Low' columns.

    Adds columns to df:
    - 'high_20': Highest high over the previous 20 days, adjusting for early rows.
    - 'low_20': Lowest low over the previous 20 days, adjusting for early rows.
    - 'high_10': Highest high over the previous 10 days, adjusting for early rows.
    - 'low_10': Lowest low over the previous 10 days, adjusting for early rows.
    - 'long_entry': Signal for entering a long position.
    - 'long_exit': Signal for exiting a long position.
    - 'short_entry': Signal for entering a short position.
    - 'short_exit': Signal for exiting a short position.
    """
    df['datetime'] = pd.to_datetime(df['timeframe'], unit='ms')
    # Calculate rolling max/min for the required windows with min_periods=1
    df['high_20'] = df['H'].rolling(window=TURTLE_ENTRY_DAYS, min_periods=1).max()
    df['low_20'] = df['L'].rolling(window=TURTLE_ENTRY_DAYS, min_periods=1).min()
    df['high_10'] = df['H'].rolling(window=TURTLE_EXIT_DAYS, min_periods=1).max()
    df['low_10'] = df['L'].rolling(window=TURTLE_EXIT_DAYS, min_periods=1).min()

    # Entry signals
    df['long_entry'] = df['H'] > df['high_20'].shift(1)
    df['short_entry'] = df['L'] < df['low_20'].shift(1)

    # Exit signals
    df['long_exit'] = df['L'] < df['low_10'].shift(1)
    df['short_exit'] = df['H'] > df['high_10'].shift(1)

    return df


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

            # If there are no records matching the filters, set the values to 0.0
            strategy_pl = 0.0 if strategy_pl is None else float(strategy_pl)
            total_pl = 0.0 if total_pl is None else float(total_pl)

        return last_closed_position_pl, round(strategy_pl, 1), round(total_pl, 1)

    def log_total_pl(self):
        last_closed_pl, strategy_pl, total_pl = self.get_pl()
        msg = (f'\n==={self._exchange.market} on {self.strategy_settings.exchange_id}===\n'
               f'strategy id: {self.strategy_settings.id}\n'
               f'last trade P/L: {last_closed_pl[0]}$, {last_closed_pl[1]}%\n'
               f'strategy P/L: {strategy_pl}$\n'
               f'Total P/L: {total_pl}$\n'
               f'Total balance: {round(self._exchange.total_balance, 0)}$')
        _logger.info(msg)
        _notifier.info(msg)

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
        n_days_ago = datetime.now() - timedelta(days=self.strategy_settings.buffer_days)
        since_timestamp_ms = int(n_days_ago.timestamp() * 1000)

        if testing_file_path:
            ohlc = pd.read_csv(testing_file_path)
        else:
            ohlc = self._exchange.fetch_ohlc(since=since_timestamp_ms, timeframe=self.strategy_settings.timeframe)

        ohlc = calculate_atr(ohlc, period=ATR_PERIOD)
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

        if action == 'close':
            order_object.closed_positions = self.opened_positions_ids
            order_object.pl, order_object.pl_percent = self.calculate_pl(order_object)

        try:
            save_json_to_file(order_object, f"order_{order_object.id}")
        except Exception as exc:
            _logger.error(f"Cannot save json file, skipp. {exc}")
            _notifier.error(f"Cannot save json file, skipp. {exc}")

        self.commit_order_to_db(order_object)

    def entry_position(self, action):
        self._exchange.fetch_balance()
        free_balance = self._exchange.free_balance
        total_balance = self._exchange.total_balance
        free_balance = self.recalc_limited_free_entry_balance(free_balance, total_balance)

        trade_risk_cap = free_balance * TRADE_RISK_ALLOCATION
        amount = (trade_risk_cap /
                  (self.strategy_settings.stop_loss_atr_multipl *
                   self.curr_market_conditions.atr_20 *
                   self.curr_market_conditions.atr_period_ratio)
                  )
        _logger.info(f"Amount before rounding: {amount}")
        amount = get_adjusted_amount(amount, self._exchange.amount_precision)
        _logger.info(f"Amount after precision rounding: {amount}")

        cost = self.curr_market_conditions.C * amount
        if cost < self._exchange.min_cost or cost > free_balance:
            _logger.warning(f"Cost {cost} is lower than "
                            f"min cost {self._exchange.min_cost} "
                            f"or higher than free_balance {free_balance}"
                            f"SKIPPING ticker")
            return

        # each exchange has its own contract size
        # for example DOGE on binance = 1 but on mexc = 100
        # if we want to buy 100 of doge on binance = 100 contracts
        # but on mexc = 1 contract
        amount = amount / self._exchange.contract_size
        if amount < self._exchange.min_amount:
            _logger.warning(f"Cost {cost} is lower than "
                            f"min cost {self._exchange.min_cost} "
                            f"or higher than free_balance {free_balance}"
                            f"SKIPPING ticker")
            amount = self._exchange.min_amount

        _logger.info(f"Amount: {amount}"
                     f"Contract size: {self._exchange.contract_size}"
                     f"Amount of contracts: {amount}")

        order = self._exchange.order(action, amount)

        if order:
            self.save_order(order, action)

        msg = (f"Exchange: {self._exchange.exchange_id}\n"
               f"strategy id: {self.strategy_settings.id}\n"
               f"Entered {self.strategy_settings.ticker}\n"
               f"Position: {action}\n"
               f"Amount: {amount} (amount can be different based on exchange handling contract size)\n"
               f"Cost: {cost} (amount can be different based on exchange handling contract size)\n")
        _logger.info(msg)
        _notifier.info(msg)

    def exit_position(self):
        action = 'close'
        order = self._exchange.order(action)
        if order:
            self.save_order(order, action, position_status='closed')
            self.update_closed_orders()
            self.log_total_pl()

    def process_opened_position(self):
        _logger.info('Processing opened positions')

        curr_mar_cond = self.curr_market_conditions
        last_stop_loss = self.last_opened_position.stop_loss_price
        long_pyramid_price, short_pyramid_price = self.get_atr_for_pyramid()

        # check if number of pyramid trade is over limit
        pyramid_stop = self.n_of_opened_positions > self.strategy_settings.pyramid_entry_limit

        if self.last_opened_position.is_long():
            # exit position
            if curr_mar_cond.long_exit:
                _logger.info('Exiting long position/s')
                self.exit_position()
            # add to position -> pyramiding
            elif curr_mar_cond.C >= long_pyramid_price and not pyramid_stop:
                _logger.info(f'Adding to long position -> pyramid')
                self.entry_position('long')
            # exit position -> stop loss
            elif curr_mar_cond.C <= last_stop_loss:
                _logger.info('Initiating long stop-loss')
                self.exit_position()
            else:
                _logger.info('Staying in position '
                             '-> no condition for opened position is met')
        else:
            # exit position
            if curr_mar_cond.short_exit:
                _logger.info('Exiting short position/s')
                self.exit_position()
            # add to position -> pyramiding
            elif curr_mar_cond.C <= short_pyramid_price and not pyramid_stop:
                _logger.info(f'Adding to short position -> pyramid')
                self.entry_position('short')
            # exit position -> stop loss
            elif curr_mar_cond.C >= last_stop_loss:
                _logger.info('Initiating short stop-loss')
                self.exit_position()
            else:
                _logger.info('Staying in position '
                             '-> no condition for opened position is met')

    def trade(self):
        if self.opened_positions is None:
            curr_cond = self.curr_market_conditions
            # entry long
            if curr_cond.long_entry and not curr_cond.long_exit:  # safety
                _logger.info('Long cond is met -> entering long position')
                self.entry_position('long')
            # entry short
            elif curr_cond.short_entry and not curr_cond.short_exit:  # safety
                _logger.info('Short cond is met -> entering short position')
                self.entry_position('short')
            # do nothing
            else:
                _logger.info('No opened positions and no condition is met for entry -> SKIPPING')
        # work with opened position
        else:
            self.process_opened_position()

    def close_position(self):
        if self.opened_positions is not None:
            self.exit_position()
