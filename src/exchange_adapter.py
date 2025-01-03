import asyncio
import logging
import traceback
from copy import deepcopy
from datetime import datetime, timedelta

import ccxt
import ccxt.async_support as accxt  # Async CCXT
import pandas as pd
from retrying import retry
from slack_bot.notifications import SlackNotifier

from src.config import SLACK_URL, LEVERAGE, app_config

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
    global_params = {'leverage': LEVERAGE}

    def __init__(self, exchange_id: str, sub_account_id: str = None, market: str = None):
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, self.exchange_id)

        # Start with a deep copy of the exchange configuration
        base_config = deepcopy(app_config.EXCHANGES[exchange_id])

        # Apply sub-account keys if provided
        if sub_account_id:
            sub_account_config = base_config.get('sub_accounts', {}).get(sub_account_id)
            if not sub_account_config:
                raise ValueError(f"Sub-account '{sub_account_id}' not found for exchange '{exchange_id}'")
            # Explicitly update only the API credentials
            base_config['apiKey'] = sub_account_config['apiKey']
            base_config['secret'] = sub_account_config['secret']

        # Remove sub-accounts key to avoid passing it to the exchange object
        base_config.pop('sub_accounts', None)

        self._exchange_config = base_config
        self._exchange = exchange_class(self._exchange_config)

        self._base_currency = self._exchange.base_currency
        self._market = f"{market}/{self._base_currency}"
        self.market_futures = f"{self._market}:{self._base_currency}"

        self.markets = None
        self._open_positions = None
        self._open_orders = None
        self.balance = None
        self._trade_history = None

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
    def contract_size(self):
        try:
            return self.market_info['contract_size']
        except KeyError:
            return self.market_info.get('contractSize', 1)

    @property
    def free_balance(self):
        if not self.balance:
            self.fetch_balance()
        if self.balance:
            binance_bnfcr = self.balance['free'].get('BNFCR', 0)
            balance_base = self.balance['free'].get(self._base_currency, 0)
            free = balance_base + binance_bnfcr
            return free
        return 0

    @property
    def total_balance(self):
        if not self.balance:
            self.fetch_balance()
        if self.balance:
            binance_bnfcr = self.balance['total'].get('BNFCR', 0)
            balance_base = self.balance['total'].get(self._base_currency, 0)
            total = balance_base + binance_bnfcr
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

    @property
    def default_trading_type(self):
        return self._exchange_config['options']['defaultType']

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def fetch_ohlc(self, ticker=None, days=100, timeframe: str = '4h', limit=500):
        if not ticker:
            ticker = self.market_futures

        n_days_ago = datetime.now() - timedelta(days=days)
        since = int(n_days_ago.timestamp() * 1000)

        all_candles = []  # Store all fetched candles here
        candles_df = pd.DataFrame()

        # Convert buffer_days to the timestamp corresponding to that number of days ago
        end_time = datetime.now().timestamp() * 1000  # Current timestamp in milliseconds

        while since < end_time:
            # Fetch candles from the exchange with a limit on the number of records
            candles = self._exchange.fetchOHLCV(ticker, timeframe=timeframe, since=since, limit=limit)

            # If no more candles are returned, break the loop
            if not candles:
                break

            # Append the fetched candles to the list
            all_candles.extend(candles)

            # Update 'since' to the timestamp of the last candle to fetch the next batch
            since = candles[-1][0] + 1  # Increment to avoid overlapping with the last candle

            # Check if the newly fetched data brings us to the present (end_time)
            if candles[-1][0] >= end_time:
                break

        # Convert the list of candles to a DataFrame
        if all_candles:
            candles_df = pd.DataFrame(all_candles, columns=['timeframe', 'O', 'H', 'L', 'C', 'V'])
            candles_df['datetime'] = pd.to_datetime(candles_df['timeframe'], unit='ms')

        return candles_df

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def get_open_interest_hist(self, timeframe: str = '4h', since: datetime.timestamp = None):
        try:
            # Check if the exchange supports the fetch_open_interest method
            if hasattr(self._exchange, 'fetch_open_interest_history'):
                # Fetch open interest for the specified market symbol
                open_interest_data = self._exchange.fetch_open_interest_history(
                    self.market_futures, timeframe=timeframe, since=since
                )
                return open_interest_data
            else:
                raise ccxt.NotSupported(f"{self._exchange.id} does not support fetching open interest.")

        except ccxt.NotSupported as e:
            # Handle the NotSupported error if the method is not supported by the exchange
            _logger.error(f"Error: open interest not supported: {e}")
            return None

        except Exception as e:
            # Handle any other exceptions that may occur
            _logger.warning(f"An unexpected error occurred: {e}")
            return None

    def get_funding_rate(self):
        try:
            funding_rate = self._exchange.fetch_funding_rate(symbol=self.market_futures)
            if funding_rate:
                _logger.info(
                    f"Fetched {len(funding_rate)} funding rate history records for {self.market_futures}")
                return funding_rate
            else:
                _logger.info(f"No funding rate history found for {self.market_futures}")
                return None
        except Exception as e:
            _logger.error(f"Error fetching funding rate history: {str(e)}")
            return None

    def get_funding_rate_history(self):
        try:
            funding_rate_history = self._exchange.fetch_funding_rate_history(symbol=self.market_futures)
            if funding_rate_history:
                _logger.info(
                    f"Fetched {len(funding_rate_history)} funding rate history records for {self.market_futures}")
                return funding_rate_history
            else:
                _logger.info(f"No funding rate history found for {self.market_futures}")
                return None
        except Exception as e:
            _logger.error(f"Error fetching funding rate history: {str(e)}")
            return None

    @staticmethod
    def aggregate_funding_rates(funding_rate_history, period='4H'):
        # Convert funding rate history into a pandas DataFrame
        df = pd.DataFrame(funding_rate_history)

        # Convert the timestamp into a datetime object
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Extract only the relevant numerical columns (e.g., 'fundingRate')
        if 'fundingRate' in df.columns:
            # Set 'datetime' as the index, resample by the desired period, and calculate the mean funding rate
            df_resampled = df[['datetime', 'fundingRate']].set_index('datetime').resample(period).mean()

            # Reset the index to bring 'datetime' back as a column
            df_resampled = df_resampled.reset_index()

            return df_resampled
        else:
            print("Funding rate data does not have the expected 'fundingRate' column.")
            return None

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
    def get_close_price(self):
        _logger.info(f"getting close price")
        return self._exchange.fetch_ticker(symbol=self.market_futures)['close']

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def fetch_order(self, order_id):
        try:
            return self._exchange.fetch_order(order_id, self.market_futures)
        except ccxt.errors.NotSupported:
            return self._exchange.fetchOpenOrder(order_id, self.market_futures)

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def get_opened_positions(self):
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
            self._open_positions = open_position

        return open_positions

    def get_open_positions_all(self):
        """
        fetch open positions for the current market.
        """
        open_positions = self._exchange.fetch_positions()
        return open_positions

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def get_opened_orders(self):
        """Fetches open (unfilled) orders for the specified market."""
        _logger.info(f"Fetching open orders for {self.market_futures}")

        open_orders = self._exchange.fetch_open_orders(symbol=self.market_futures)

        if open_orders:
            self._open_orders = open_orders
            _logger.info(f"Found {len(open_orders)} open orders.")
        else:
            self._open_orders = None
            _logger.info(f"No open orders found.")

        return open_orders

    def get_opened_orders_all(self, symbols):
        raise NotImplemented

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def get_trade_history(self):
        """Fetches trade history (executed trades) for the specified market."""
        _logger.info(f"Fetching trade history for {self.market_futures}")

        trade_history = self._exchange.fetch_my_trades(symbol=self.market_futures)

        if trade_history:
            self._trade_history = trade_history
            _logger.info(f"Found {len(trade_history)} trades.")
        else:
            _logger.info("No trades found.")

        return trade_history

    @staticmethod
    def process_last_trade_with_sl_tp(trade_history):
        """
        Processes the last trade in the trade history and determines if it closed a position.
        Uses 'stopOrderType' and 'createType' to identify if a stop-loss or take-profit was hit.

        Args:
            trade_history (list): A list of trades from the exchange.

        Returns:
            dict: Last closed trade information in the required format.
        """

        if not trade_history:
            return {"last_closed_trade": None}

        last_closed_trade = {}
        total_buy = 0.0  # Total buy amount
        total_sell = 0.0  # Total sell amount

        for trade in trade_history:
            side = trade['side']  # 'buy' or 'sell'
            amount = float(trade['amount'])  # Amount of the trade
            price = float(trade['price'])  # Trade price
            timestamp = trade['timestamp']
            stop_order_type = trade['info'].get('stopOrderType')  # Check stop order type if exists
            create_type = trade['info'].get(
                'createType')  # Check create type (e.g., CreateByStopLoss, CreateByTakeProfit)

            # Accumulate buy and sell amounts
            if side == 'buy':
                total_buy += amount
            elif side == 'sell':
                total_sell += amount

            # If the trade was triggered by a stop-loss or take-profit, mark it as such
            hit_stop_loss = create_type == 'CreateByStopLoss' or stop_order_type == 'StopLoss'
            hit_take_profit = create_type == 'CreateByTakeProfit' or stop_order_type == 'TakeProfit'

            # If total buy equals total sell, the position is closed
            if total_buy == total_sell:
                # Calculate profit or loss based on the entry price (first trade)
                entry_price = trade_history[0]['price']
                profit_or_loss = (price - entry_price) * total_sell if side == 'sell' \
                    else (entry_price - price) * total_buy
                outcome = 'profit' if profit_or_loss > 0 else 'loss'

                # Construct the last closed trade dictionary
                last_closed_trade = {
                    "action": "close",
                    "amount": total_sell if side == 'sell' else total_buy,
                    "timestamp": timestamp,
                    "hit_stop_loss": hit_stop_loss,
                    "hit_take_profit": hit_take_profit,
                    "outcome": outcome,
                    "price": price,
                    "profit_or_loss": round(profit_or_loss, 2),
                    "side": side
                }

        return last_closed_trade if last_closed_trade else None

    @property
    def open_position_amount(self):
        return self._open_positions['contracts']

    @property
    def open_position_side(self):
        if self._open_positions:
            return POSITIONS_MAPPING.get(self._open_positions['side'])
        return

    @property
    def open_position_equity(self):
        return self._open_positions['initialMargin'] + self._open_positions['unrealizedPnl']

    def assert_side(self, side):
        assert side != self.open_position_side, (
            f"There's already "
            f"opened position with"
            f" the same side: {self.open_position_side},"
            f"\n{self._open_positions}"
        )

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def cancel_order(self, order_id: str, symbol=None):
        if not symbol:
            symbol = self.market_futures
        return self._exchange.cancel_order(order_id, symbol)

    def update_sl_tp(self, take_profit=None, stop_loss=None):
        """
        Update Take-Profit (TP) and Stop-Loss (SL) for existing positions using edit_order.

        :param take_profit: Take-Profit price
        :param stop_loss: Stop-Loss price
        """
        if not self._open_orders:
            _logger.warning("No open orders. Skipping TP/SL update.")
            return

        # Separate SL and TP orders
        opened_sl_orders = [
            order for order in self._open_orders
            if order['info'].get('stopOrderType') == 'StopLoss'
        ]
        opened_tp_orders = [
            order for order in self._open_orders
            if order['info'].get('stopOrderType') == 'TakeProfit'
        ]

        # Update Stop-Loss Order
        if stop_loss:
            if len(opened_sl_orders) > 1:
                _logger.warning("Multiple Stop-Loss orders found. Skipping update.")
                return
            if opened_sl_orders:
                sl_order = opened_sl_orders[0]
                sl_order_id = sl_order['id']
                side = sl_order['side']

                try:
                    params = {'stopLossPrice': stop_loss}
                    edited_sl_order = self._exchange.edit_order(
                        id=sl_order_id,
                        symbol=self.market_futures,
                        type='market',
                        side=side,
                        params=params
                    )
                    _logger.info(f"Stop-Loss updated to {stop_loss}: {edited_sl_order}")
                except Exception as e:
                    _logger.error(f"Failed to update Stop-Loss: {e}")

        # Update Take-Profit Order
        if take_profit:
            if len(opened_tp_orders) > 1:
                _logger.warning("Multiple Take-Profit orders found. Skipping update.")
                return
            if opened_tp_orders:
                tp_order = opened_tp_orders[0]
                tp_order_id = tp_order['id']
                side = tp_order['side']

                try:
                    params = {'takeProfitPrice': take_profit}
                    edited_tp_order = self._exchange.edit_order(
                        id=tp_order_id,
                        symbol=self.market_futures,
                        type='market',
                        side=side,
                        params=params
                    )
                    _logger.info(f"Take-Profit updated to {take_profit}: {edited_tp_order}")
                except Exception as e:
                    _logger.error(f"Failed to update Take-Profit: {e}")

        if not (stop_loss or take_profit):
            _logger.info("No updates for TP or SL were requested.")

    def edit_order(self, ordr_id, price=None, stop_loss=None, take_profit=None):
        order = [
            order for order in self._open_orders
            if order['id'] == ordr_id
        ]
        if not order:
            return
        else:
            order = order[0]

        side = order['side']
        params = {}
        try:
            if take_profit:
                params['takeProfitPrice'] = take_profit
            if stop_loss:
                params['stopLossPrice'] = stop_loss

            edited_tp_order = self._exchange.edit_order(
                id=ordr_id,
                symbol=self.market_futures,
                type='market',
                side=side,
                price=price,
                params=params
            )
            _logger.info(f"Take-Profit updated to {take_profit}: {edited_tp_order}")
        except Exception as e:
            _logger.error(f"Failed to update Take-Profit: {e}")

    def _set_leverage(self, leverage):
        try:
            _logger.info(f"Setting leverage to {leverage} for {self.market_futures}")
            self._exchange.set_leverage(leverage, self.market_futures)
        except Exception:
            _logger.info(f"Leverage set up not supported on this exchange! {traceback.format_exc()}")

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1500)
    def enter_position(self, side, amount, limit_price=None, stop_loss=None, take_profit=None, leverage=None):
        _logger.info(f"entering {str.upper(side)} position")
        if not leverage:
            leverage = self.global_params.get('leverage', 1)
            params = self.global_params
        else:
            self.global_params['leverage'] = leverage
            params = self.global_params

        # Set stop-loss and take-profit if provided
        if stop_loss:
            params['stopLoss'] = {'triggerPrice': stop_loss}
        if take_profit:
            params['takeProfit'] = {'triggerPrice': take_profit}

        order_type = 'market'
        if limit_price:
            order_type = 'limit'

        try:
            # self.opened_position()
            # self.assert_side(side)
            #
            # if self.open_position_side:
            #     _logger.info(f"There is already one open position")
            #     self.close_position()
            self._set_leverage(leverage)

            _logger.info(f"creating order: {side}, "
                         f"order_type: {order_type}, "
                         f"amount: {amount}, "
                         f"limit_price: {limit_price}, "
                         f"params: {self.global_params}")
            order = self._exchange.create_order(
                symbol=self.market_futures,
                type=order_type,
                side=side,
                amount=amount,
                params=params,
                price=limit_price
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

        except Exception:
            msg = f"{self._exchange.id} close_position failed with: {traceback.format_exc()}"
            _logger.error(msg)
            raise

    @retry(retry_on_exception=retry_if_network_error,
           stop_max_attempt_number=5,
           wait_exponential_multiplier=1000)
    def close_position(self, amount: float = None):
        _logger.info(f"closing position")

        params = {'reduceOnly': True}
        try:
            self.get_opened_positions()

            if not self.open_position_side:
                _logger.warning(f"no open position to close: {self.open_position_side}")
                _notifier.warning(f"no open position to close: {self.open_position_side}")
                return {"msg": "Nothing to close"}

            side = 'buy' if self.open_position_side == 'sell' else 'sell'

            amount = self.open_position_amount if not amount else amount
            _logger.info(f"creating order: {side}, "
                         f"amount: {amount}, "
                         f"params: {params}")
            order = self._exchange.create_order(
                symbol=self.market_futures,
                type='market',
                side=side,
                amount=amount,
                params=params
            )

            # _notifier.info(f"{self.market_futures} position CLOSED, amount: {amount}")
            return order

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            msg = (f"{self._exchange.id} close_position failed "
                   f"due to a Network or Exchange error: {e}")
            _logger.error(msg)
            raise

        except Exception:
            msg = f"{self._exchange.id} close_position failed with: {traceback.format_exc()}"
            _logger.error(msg)
            raise

    def order(self,
              action_key,
              amount: float = 0,
              limit_price: float = None,
              stop_loss: float = None,
              take_profit: float = None,
              leverage: int = None):

        _actions = {
            'long': {'action': self.enter_position, 'side': 'buy'},
            'short': {'action': self.enter_position, 'side': 'sell'},
            'close': {'action': self.close_position}
        }

        position = _actions.get(action_key)
        position_order = position['action']
        side = position.get('side', None)

        if side:
            return position_order(side, amount, limit_price, stop_loss, take_profit, leverage)
        return position_order(amount)


# class MexcExchange(BaseExchangeAdapter):
#     def enter_position(self, side, amount):
#         leverage = self.global_params.get('leverage', 1)
#         position_type = 1 if side == 'buy' else 2
#         open_type = 1  # Isolated margin
#
#         _logger.info(f"Setting leverage for MEXC: leverage={leverage}, "
#                      f"openType={open_type}, positionType={position_type}")
#         self._exchange.set_leverage(
#             leverage,
#             self.market_futures,
#             {'openType': open_type, 'positionType': position_type}
#         )
#
#         params = {
#             'vol': amount,
#             'leverage': leverage,
#             'side': position_type,
#             'type': 5,
#             'openType': open_type,
#             'positionMode': 1
#         }
#
#         order = self._exchange.create_order(
#             symbol=self.market_futures,
#             type='market',
#             side=side,
#             amount=amount,
#             params=params
#         )
#
#         _logger.info(f"Order created on MEXC: {order}")
#         return order


class AsyncExchangeAdapter(BaseExchangeAdapter):
    def __init__(self, exchange_id: str, sub_account_id: str = None, market: str = None):
        super().__init__(exchange_id, sub_account_id, market)
        exchange_class = getattr(accxt, self.exchange_id)
        self._exchange = exchange_class(self._exchange_config)  # Replace with async exchange

    async def load_exchange(self, force_refresh=True):
        if force_refresh or not self._exchange.markets:
            _logger.info(f"Loading markets on {self._exchange.id}")
            self.markets = await self._exchange.load_markets(True)
        _logger.info("Markets loaded successfully")

    async def fetch_ohlc(self, ticker=None, days=100, timeframe: str = '4h', limit=500):
        """
        Asynchronously fetch OHLC data for a specific ticker.
        """
        if not ticker:
            ticker = self.market_futures

        n_days_ago = datetime.now() - timedelta(days=days)
        since = int(n_days_ago.timestamp() * 1000)

        all_candles = []
        end_time = datetime.now().timestamp() * 1000

        while since < end_time:
            candles = await self._exchange.fetch_ohlcv(ticker, timeframe=timeframe, since=since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1

        if all_candles:
            candles_df = pd.DataFrame(all_candles, columns=['time', 'O', 'H', 'L', 'C', 'V'])
            candles_df['datetime'] = pd.to_datetime(candles_df['time'], unit='ms')
            candles_df['ticker'] = ticker.split('/')[0]
            return candles_df
        return pd.DataFrame()

    async def get_open_positions_all(self):
        """
        Asynchronously fetch open positions for the current market.
        """
        _logger.info("getting open positions")
        open_positions = await self._exchange.fetch_positions()
        return open_positions

    async def async_fetch_ohlc(self, tickers, days=220, timeframe='1d', limit=500):
        """
        Fetch OHLC data and open positions for multiple tickers concurrently.
        """
        ohlc_tasks = [self.fetch_ohlc(ticker, days, timeframe, limit) for ticker in tickers]

        ohlc_results = await asyncio.gather(asyncio.gather(*ohlc_tasks))
        return ohlc_results

    async def fetch_balance(self):
        _logger.info(f"getting balance")
        self.balance = await self._exchange.fetch_balance()

    async def get_opened_orders(self, symbol=None):
        """
        Fetch open orders for a single symbol asynchronously.
        :param symbol: Symbol (ticker) to fetch open orders for
        :return: List of orders
        """
        if not symbol:
            symbol = self._exchange.markets
        try:
            return await self._exchange.fetch_open_orders(symbol)
        except Exception as e:
            print(f"Error fetching open orders for {symbol}: {e}")
            return []

    async def get_opened_orders_all(self, symbols):
        """
        Fetch open orders for multiple symbols concurrently.

        :param symbols: List of symbols (tickers) to fetch open orders for
        :return: List of orders
        """
        tasks = [self.get_opened_orders(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results into a single list, filtering out None or empty lists
        orders = [
            order for orders_per_symbol in results if isinstance(orders_per_symbol, list)
            for order in orders_per_symbol
        ]
        return orders

    async def close(self):
        """
        Properly close the exchange connection.
        """
        await self._exchange.close()
