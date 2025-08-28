import logging
import sys
import traceback

import click
import numpy as np
import pandas as pd
from jnd_utils.log import init_logging
from slack_bot.notifications import SlackNotifier

from config import SLACK_URL, LLM_TRADER_SLACK_URL
from exchange_adapter import BaseExchangeAdapter
from src.exchange_factory import ExchangeFactory
from src.llm_trader import LlmTrader
from src.ticker_picker import LlmTickerPicker, TickerPicker
from src.turtle_trader import ShitTrader
from src.utils.turtle_back_test import turtle_back_test
from src.utils.utils import load_strategy_settings, save_total_balance, process_new_shit_trader_strategy
from turtle_trader import TurtleTrader

_logger = logging.getLogger(__name__)
_notifier = SlackNotifier(url=SLACK_URL, username='turtle_trader')
_notifier_llm = SlackNotifier(url=LLM_TRADER_SLACK_URL, username='llm_trader')


@click.group(chain=True)
def cli():
    pass


@cli.command()
@click.option('-exch', '--exchange_id', type=str, default='binance')
def log_pl(exchange_id):
    _notifier.info("===Just logging P/L ===\n\n")
    exchange: BaseExchangeAdapter = ExchangeFactory.get_exchange(exchange_id)
    strategy_settings = load_strategy_settings(exchange_id)
    for strategy in strategy_settings:
        exchange.market = f"{strategy.ticker}"
        TurtleTrader(exchange=exchange, strategy_settings=strategy).log_total_pl()


@cli.command(help='close position manually')
@click.option('-exch', '--exchange_id', type=str, default='bybit')
@click.option('-si', '--strategy_ids', type=str)
def close_position(exchange_id, strategy_ids):
    strategy_ids = [int(sid) for sid in strategy_ids.split(',')]
    _logger.info(f"\n============== CLOSING POSITION {strategy_ids} ==============\n")
    agent_map = {
        'turtle_trader': TurtleTrader,
        'shit_trader': ShitTrader
    }
    try:
        strategy_settings = load_strategy_settings(exchange_id)
        if not strategy_settings:
            return _logger.info("No active strategy found, skipping")

        for strategy in strategy_settings:
            if strategy.id in strategy_ids:
                exchange: BaseExchangeAdapter = ExchangeFactory.get_exchange(
                    exchange_id,
                    sub_account_id=strategy.sub_account_id
                )
                exchange.load_exchange()
                trader_obj = agent_map.get(strategy.agent_id)
                exchange.market = f"{strategy.ticker}"
                trader = trader_obj(exchange, strategy)
                trader.close_position()
    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


@cli.command(help='run Turtle trading bot')
@click.option('-exch', '--exchange_id', type=str, default='bybit')
def trade(exchange_id):
    _logger.info("\n============== STARTING TRADE SESSION ==============\n")
    try:
        strategy_settings = load_strategy_settings(exchange_id, agent_id='turtle_trader')
        if not strategy_settings:
            return _logger.info("No active strategy found, skipping")

        async_exchange_adapter = ExchangeFactory.get_async_exchange(
            exchange_id, sub_account_id=strategy_settings[0].sub_account_id
        )
        ticker_picker = TickerPicker(async_exchange_adapter, strategy_settings)
        strategy_settings_filtered = ticker_picker.pick_tickers()
        if not strategy_settings_filtered:
            _logger.info("No entry conditions met...")
            return

        exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(
            exchange_id, sub_account_id=strategy_settings[0].sub_account_id
        )
        _logger.info(f"Initialising Turtle trader on {exchange_id}, "
                     f"tickers: {[x.ticker for x in strategy_settings]}")
        exchange_adapter.load_exchange()

        for strategy in strategy_settings_filtered:
            _logger.info(f"\n\n----------- Starting trade - {strategy.ticker}, "
                         f"strategy_id: {strategy.id}-----------")
            exchange_adapter.market = f"{strategy.ticker}"
            trader = TurtleTrader(exchange=exchange_adapter, strategy_settings=strategy)
            _logger.debug(f"Market info before trading: {exchange_adapter.market_info}")
            trader.trade()

    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}", echo='here')
        sys.exit(1)


@cli.command(help='run Turtle trading bot')
@click.option('-exch', '--exchange_id', type=str, default='bybit')
@click.option('-e', '--enter', is_flag=True, default=False)
@click.option('-t', '--ticker', type=str)
@click.option('-f', '--timeframe', type=str, default='4h')
@click.option('-s', '--side', type=str)
@click.option('-sb', '--standby-mode', is_flag=True, default=False)
@click.option('-r', '--risk', type=float, default=None)
def shit_trade(exchange_id, enter, ticker, timeframe, side, standby_mode, risk):
    _logger.info("\n============== STARTING TRADE SESSION ==============\n")
    try:
        if enter:
            process_new_shit_trader_strategy(exchange_id, ticker, timeframe, side, standby_mode)

        strategy_settings = load_strategy_settings(exchange_id, agent_id='shit_trader')
        if not strategy_settings:
            return _logger.info("No active strategy found, skipping")

        exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(
            exchange_id, sub_account_id=strategy_settings[0].sub_account_id
        )
        _logger.info(f"Initialising Turtle trader on {exchange_id}, "
                     f"tickers: {[x.ticker for x in strategy_settings]}")
        exchange_adapter.load_exchange()

        for strategy in strategy_settings:
            _logger.info(f"\n\n----------- Starting trade - {strategy.ticker}, "
                         f"strategy_id: {strategy.id}-----------")
            exchange_adapter.market = f"{strategy.ticker}"
            trader = ShitTrader(exchange=exchange_adapter, strategy_settings=strategy, trade_risk=risk)
            _logger.debug(f"Market info before trading: {exchange_adapter.market_info}")
            trader.trade()

    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}", echo='here')
        sys.exit(1)


@cli.command(help='run LLM trading bot')
@click.option('-exch', '--exchange_id', type=str, default='bybit')
@click.option('-t', '--tickers', type=str, default=None)
@click.option('-tp', '--ticker-picker', is_flag=True, default=False)
def llm_trade(exchange_id, tickers, ticker_picker):
    _logger.info("\n============== STARTING LLM TRADE SESSION ==============\n")
    if tickers:
        tickers = tickers.split(',')
    try:
        strategy_settings = load_strategy_settings(exchange_id, 'llm_trader')
        if not strategy_settings:
            return _logger.info("No active strategy found, skipping")
        _logger.info(f"Initialising LLM trader on {exchange_id}, "
                     f"tickers: {[x.ticker for x in strategy_settings]}")

        if tickers:
            strategy_settings_filtered = [
                strategy for strategy in strategy_settings
                if strategy.ticker in tickers
            ]
        elif ticker_picker:
            # TickerPicker
            async_exchange_adapter = ExchangeFactory.get_async_exchange(
                exchange_id, sub_account_id=strategy_settings[0].sub_account_id
            )
            llm_ticker_picker = LlmTickerPicker(async_exchange_adapter, strategy_settings)
            strategy_settings_filtered = llm_ticker_picker.llm_pick_tickers()
        else:
            strategy_settings_filtered = strategy_settings

        # Llm Trader for each picked ticker
        exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(
            exchange_id, sub_account_id=strategy_settings[0].sub_account_id
        )
        exchange_adapter.load_exchange()
        for strategy in strategy_settings_filtered:
            _logger.info(f"\n\n----------- Starting trade - {strategy.ticker}, "
                         f"strategy_id: {strategy.id}-----------")
            exchange_adapter.market = f"{strategy.ticker}"
            trader = LlmTrader(exchange_adapter, strategy)
            exchange_adapter.fetch_balance()
            trader.trade()

        save_total_balance(
            exchange_id=exchange_id,
            total_balance=exchange_adapter.total_balance,
            sub_account_id=strategy_settings[0].sub_account_id
        )

    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier_llm.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


@cli.command(help='run LLM trading bot')
@click.option('-exch', '--exchange_id', type=str, default='bybit')
def back_test(exchange_id):
    _logger.info("\n============== BACKTEST ==============\n")
    turtle_back_test(exchange_id)


@cli.command(help='run LLM trading bot')
@click.option('-exch', '--exchange_id', type=str, default='bybit')
def test_volprofile(exchange_id):
    exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(exchange_id, 'subAccount1')
    exchange_adapter.load_exchange()
    exchange_adapter.market = 'BTC'
    df = exchange_adapter.fetch_ohlc(days=10, timeframe='1d')

    # Use -3 for high/low, -2 for close/volume
    prev_high, prev_low = df.iloc[-2]['H'], df.iloc[-2]['L']
    prev_close, prev_volume = df.iloc[-2]['C'], df.iloc[-2]['V']

    # Define finer price bins
    num_bins = 100
    price_bins = np.linspace(prev_low, prev_high, num_bins)

    # Assign each closing price to a bin
    df['price_bin'] = pd.cut(df['C'], bins=price_bins, labels=False)

    # Compute volume profile
    volume_profile = df.groupby('price_bin')['V'].sum().dropna()

    # Validate total volume
    total_volume = volume_profile.sum()
    if total_volume == 0:
        raise ValueError("Total volume is zero. Check data source.")

    # Normalize volume
    volume_profile /= total_volume

    # Sort bins by volume & compute cumulative sum
    volume_profile_sorted = volume_profile.sort_values(ascending=False)
    cumulative_volume = volume_profile_sorted.cumsum()

    # Debugging Output
    print("🔍 Volume Profile (Top 10 Levels):")
    print(volume_profile_sorted.head(10))
    print("🔍 Cumulative Volume (Top 10 Levels):")
    print(cumulative_volume.head(10))

    # Accumulate volume until we reach 70%
    value_area_threshold = 0.70
    cumulative_sum, value_area_bins = 0, []

    for index, volume in volume_profile_sorted.items():
        cumulative_sum += volume
        value_area_bins.append(index)
        if cumulative_sum >= value_area_threshold:
            break

    # Ensure value_area_bins is not empty
    if not value_area_bins:
        raise ValueError("Error: No valid value area bins found. Possible issue with volume distribution.")

    # Convert to integer indices & validate range
    value_area_bins = [int(x) for x in value_area_bins]
    if min(value_area_bins) < 0 or max(value_area_bins) >= len(price_bins):
        raise ValueError(f"Index out of range: min={min(value_area_bins)}, max={max(value_area_bins)}")

    # 🔥 **Fix: Ensuring VAH > VAL**
    VAL, VAH = sorted([price_bins[min(value_area_bins)], price_bins[max(value_area_bins)]])

    # Identify POC (highest volume bin)
    POC_index = volume_profile_sorted.idxmax()
    if np.isnan(POC_index) or POC_index < 0 or POC_index >= len(price_bins):
        raise ValueError(f"Invalid POC index: {POC_index}")

    POC = price_bins[int(POC_index)]

    print(f"✅ Adjusted VAH: {VAH}, VAL: {VAL}, POC: {POC}")

    return VAH, VAL, POC

if __name__ == '__main__':
    init_logging()
    cli()
