import logging
import traceback

import ccxt
import pandas as pd
from retrying import retry
from slack_bot.notifications import SlackNotifier

from config import SLACK_URL, LEVERAGE, app_config

_notifier = SlackNotifier(url=SLACK_URL, username='Exchange adapter')
_logger = logging.getLogger(__name__)

POSITIONS_MAPPING = {
    'long': 'buy',
    'short': 'sell'
}


class NotEnoughBalanceException(Exception):
    """Balance is really low"""


def retry_if_network_error(exception):
    """Return True if we should retry (in this case when it's a NetworkError), False otherwise"""
    return isinstance(exception, (ccxt.NetworkError, ccxt.ExchangeError))


class BaseExchangeAdapter:
    params = {'leverage': LEVERAGE}

    def __init__(self, exchange_id: str, market: str = None):
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, self.exchange_id)
        self._exchange_config = app_config.EXCHANGES[exchange_id]
        self._exchange = exchange_class(self._exchange_config)

        self._base_currency = self._exchange.base_currency
        self._market = f"{market}/{self._base_currency}"
        self.market_futures = f"{self._market}:{self._base_currency}"

        self.markets = None
        self._open_position = None
        self.balance = None

    def load_exchange(self, force_refresh=True):
        if force_refresh or not self._exchange.markets:
            _logger.info(f"Loading markets on {self._exchange.id}")
            self.markets = self._exchange.load_markets(True)
        _logger.info("Markets loaded successfully")

    @property
    def market_info(self):
        _logger.debug(f"Accessing market info for {self.market_futures}: "
                      f"{self.markets.get(self.market_futures, 'Market not found')}")
        return self.markets[self.market_futures]

    @property
    def amount_precision(self):
        return self.market_info['precision']['amount']

    @property
    def min_amount(self):
        min_amount = self.market_info['limits']['amount']['min']
        return min_amount if min_amount else 0

    @property
    def min_cost(self):
        min_cost = self.market_info['limits']['cost']['min']
        return min_cost if min_cost else 0

    @property
    def free_balance(self):
        if not self.balance:
            self.fetch_balance()
        if self.balance:
            binance_bnfcr = self.balance['free'].get('BNFCR', 0)
            free = self.balance['free'][self._base_currency] + binance_bnfcr
            return free
        return 0

    @property
    def total_balance(self):
        if not self.balance:
            self.fetch_balance()
        if self.balance:
            binance_bnfcr = self.balance['total'].get('BNFCR', 0)
            total = self.balance['total'][self._base_currency] + binance_bnfcr
            return total
        return 0

    @property
    def market(self) -> str:
        return self._market

    @market.setter
    def market(self, name) -> None:
        _logger.info(f"Setting market to {name}")
        self._market = f"{name}/{self._base_currency}"
        self.market_futures = f"{self._market}:{self._base_currency}"

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def fetch_ohlc(self, since, timeframe: str = '1d'):
        candles = self._exchange.fetchOHLCV(self._market, timeframe=timeframe, since=since)
        candles_df = pd.DataFrame(candles, columns=['timeframe', 'O', 'H', 'L', 'C', 'V'])
        candles_df['datetime'] = pd.to_datetime(candles_df['timeframe'], unit='ms')
        return candles_df

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def fetch_balance(self):
        _logger.info(f"getting balance")
        self.balance = self._exchange.fetch_balance()

        # if self.free_balance and self.free_balance < min_balance:
        #     _logger.error(f"balance: {self.free_balance}$ is under minimal balance: {min_balance}$")
        #     raise NotEnoughBalanceException

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def close_price(self):
        _logger.info(f"getting close price")
        return self._exchange.fetch_ticker(symbol=self.market_futures)['close']

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def opened_position(self):
        _logger.info(f"getting open positions")

        if self.exchange_id == 'binance':
            open_positions = self._exchange.fetch_account_positions(
                symbols=[self.market_futures]
            )
        else:
            open_positions = self._exchange.fetchPositions(
                symbols=[self.market_futures]
            )

        if open_positions:
            open_position = open_positions[0]
            self._open_position = open_position

    @property
    def open_position_amount(self):
        return self._open_position['contracts']

    @property
    def open_position_side(self):
        if self._open_position:
            return POSITIONS_MAPPING.get(self._open_position['side'])
        return

    @property
    def open_position_equity(self):
        return self._open_position['initialMargin'] + self._open_position['unrealizedPnl']

    def assert_side(self, side):
        assert side != self.open_position_side, (
            f"There's already "
            f"opened position with"
            f" the same side: {self.open_position_side},"
            f"\n{self._open_position}"
        )

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def enter_position(self, side, amount):
        _logger.info(f"entering {str.upper(side)} position")
        leverage = self.params.get('leverage', 1)

        _logger.info(f"Setting leverage to {leverage} for {self.market_futures}")

        try:
            # self.opened_position()
            # self.assert_side(side)
            #
            # if self.open_position_side:
            #     _logger.info(f"There is already one open position")
            #     self.close_position()

            self._exchange.set_leverage(leverage, self.market_futures)
            _logger.info(f"creating order: {side}, "
                         f"amount: {amount}, "
                         f"params: {self.params}")
            order = self._exchange.create_order(
                symbol=self.market_futures,
                type='market',
                side=side,
                amount=amount,
                params=self.params
            )

            _logger.info(f"{str.upper(side)} {self.market} | amount: {amount}")
            return order

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            msg = (f"{self._exchange.id} enter_position failed "
                   f"due to a Network or Exchange error: {e}")
            _logger.error(f"{msg}\n{traceback.format_exc()}")
            raise

        except AssertionError as e:
            msg = f"{self._exchange.id} enter_position failed due to assert error: {e}"
            _logger.error(msg)
            _notifier.error(msg)
            raise

        except Exception as e:
            msg = f"{self._exchange.id} close_position failed with: {traceback.format_exc()}"
            _logger.error(msg)
            raise

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1000)
    def close_position(self):
        _logger.info(f"closing position")

        params = {'reduceOnly': True}
        try:
            self.opened_position()

            if not self.open_position_side:
                _logger.warning(f"no open position to close: {self.open_position_side}")
                _notifier.warning(f"no open position to close: {self.open_position_side}")
                return {"msg": "Nothing to close"}

            side = 'buy' if self.open_position_side == 'sell' else 'sell'

            _logger.info(f"creating order: {side}, "
                         f"amount: {self.open_position_amount}, "
                         f"params: {params}")
            order = self._exchange.create_order(
                symbol=self.market_futures,
                type='market',
                side=side,
                amount=self.open_position_amount,
                params=params
            )

            _notifier.info(f"order CLOSE {str.upper(side)}")
            return order

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            msg = (f"{self._exchange.id} close_position failed "
                   f"due to a Network or Exchange error: {e}")
            _logger.error(msg)
            raise

        except Exception as e:
            msg = f"{self._exchange.id} close_position failed with: {traceback.format_exc()}"
            _logger.error(msg)
            raise

    def order(self, action_key, amount: float = 0):
        _actions = {
            'long': {'action': self.enter_position, 'side': 'buy'},
            'short': {'action': self.enter_position, 'side': 'sell'},
            'close': {'action': self.close_position}
        }

        position = _actions.get(action_key)
        position_order = position['action']
        side = position.get('side', None)

        if side:
            return position_order(side, amount)
        return position_order()


class MexcExchange(BaseExchangeAdapter):
    def enter_position(self, side, amount):
        leverage = self.params.get('leverage', 1)
        position_type = 1 if side == 'buy' else 2
        open_type = 1  # Isolated margin

        _logger.info(f"Setting leverage for MEXC: leverage={leverage}, "
                     f"openType={open_type}, positionType={position_type}")
        self._exchange.set_leverage(
            leverage,
            self.market_futures,
            {'openType': open_type, 'positionType': position_type}
        )

        params = {
            'vol': amount,
            'leverage': leverage,
            'side': position_type,
            'type': 5,
            'openType': open_type,
            'positionMode': 1
        }

        order = self._exchange.create_order(
            symbol=self.market_futures,
            type='market',
            side=side,
            amount=amount,
            params=params
        )

        _logger.info(f"Order created on MEXC: {order}")
        return order
