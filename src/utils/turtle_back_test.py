import logging
import sys
import traceback
from datetime import datetime, timedelta

from src.exchange_adapter import BaseExchangeAdapter
from src.exchange_factory import ExchangeFactory
from src.model import trader_database
from src.model.turtle_model import TurtleBackTest
from src.utils.utils import calculate_atr

_logger = logging.getLogger(__name__)


def calculate_back_test_profits(data, atr_multiplier=2, initial_capital=10000, risk_per_trade=0.01):
    capital = initial_capital
    position = 0  # Current position (1 for long, -1 for short, 0 for no position)
    entry_price = 0
    position_size = 0
    results = []  # Store individual trade results

    # Calculate 20-period high/low and 10-period high/low
    data['high_20'] = data['H'].rolling(window=20).max()
    data['low_20'] = data['L'].rolling(window=20).min()
    data['high_10'] = data['H'].rolling(window=10).max()
    data['low_10'] = data['L'].rolling(window=10).min()

    # Create columns for signals
    data['long_signal'] = data['H'] > data['high_20'].shift(1)  # Long entry
    data['short_signal'] = data['L'] < data['low_20'].shift(1)  # Short entry
    data['exit_long_signal'] = data['L'] < data['low_10'].shift(1)  # Long exit
    data['exit_short_signal'] = data['H'] > data['high_10'].shift(1)  # Short exit

    # Iterate through the DataFrame row-wise
    for index, row in data.iterrows():
        if position == 0:  # No position, look for entry
            if row['long_signal']:
                position = 1
                entry_price = row['C']
                risk_amount = capital * risk_per_trade
                position_size = risk_amount / (atr_multiplier * row['atr_20'])
                stop_loss = entry_price - (row['atr_20'] * atr_multiplier)
            elif row['short_signal']:
                position = -1
                entry_price = row['C']
                risk_amount = capital * risk_per_trade
                position_size = risk_amount / (atr_multiplier * row['atr_20'])
                stop_loss = entry_price + (row['atr_20'] * atr_multiplier)
        elif position == 1:  # Long position
            if row['exit_long_signal'] or row['L'] <= stop_loss:
                profit = position_size * (row['C'] - entry_price)
                capital += profit
                results.append(profit)
                position = 0
        elif position == -1:  # Short position
            if row['exit_short_signal'] or row['H'] >= stop_loss:
                profit = position_size * (entry_price - row['C'])
                capital += profit
                results.append(profit)
                position = 0

    return {
        'pl': sum(results),
        'pl_percent': round((capital - initial_capital) / initial_capital * 100, 1),
        'final_capital': capital,
        'trades': results
    }


def turtle_back_test(exchange_id):
    _logger.info("\n============== BACKTEST ==============\n")
    timeframe = '4h'
    period_days = 90

    try:
        # Using the factory to get the correct exchange adapter
        exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(exchange_id)
        exchange_adapter.load_exchange()

        tmp_markets = exchange_adapter.markets.copy()
        for market, meta in tmp_markets.items():
            if meta['type'] != exchange_adapter.default_trading_type:
                exchange_adapter.markets.pop(market)

        n_days_ago = datetime.now() - timedelta(days=period_days)
        since_timestamp_ms = int(n_days_ago.timestamp() * 1000)

        with trader_database.session_manager() as session:

            for market, meta in exchange_adapter.markets.items():
                exchange_adapter.market = meta['base']

                # ticker_saved = session.query(TurtleBackTest.ticker).filter(
                #    TurtleBackTest.ticker == exchange_adapter.market,
                #    TurtleBackTest.exchange_id == exchange_adapter.exchange_id
                # ).first()
                # if not ticker_saved:

                ohlc = exchange_adapter.fetch_ohlc(since=since_timestamp_ms, timeframe=timeframe)
                if ohlc.empty:
                    continue
                ohlc = calculate_atr(ohlc, period=20)
                result = calculate_back_test_profits(ohlc)

                obj = TurtleBackTest(
                    exchange_id=exchange_id,
                    ticker=exchange_adapter.market,
                    init_capital=10_000,
                    pl=result['pl'],
                    pl_percent=result['pl_percent'],
                    final_capital=result['final_capital'],
                    trades=result['trades'],
                    timeframe=timeframe,
                    period_days=period_days
                )

                session.add(obj)
                session.commit()
                _logger.info(f"{exchange_adapter.market} back test saved.")

    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)
