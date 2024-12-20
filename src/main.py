import logging
import sys
import traceback

import click
from jnd_utils.log import init_logging
from slack_bot.notifications import SlackNotifier

from config import SLACK_URL, LLM_TRADER_SLACK_URL
from exchange_adapter import BaseExchangeAdapter
from src.exchange_factory import ExchangeFactory
from src.llm_trader import LlmTrader
from src.ticker_picker import LlmTickerPicker, TickerPicker
from src.utils.turtle_back_test import turtle_back_test
from src.utils.utils import load_strategy_settings
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
        TurtleTrader(exchange, strategy).log_total_pl()


@cli.command(help='close position manually')
@click.option('-exch', '--exchange', type=str, default='binance')
@click.option('-si', '--strategy_id', type=int)
def close_position(exchange, strategy_id):
    _logger.info(f"\n============== CLOSING POSITION {strategy_id} ==============\n")
    try:
        strategy_settings = load_strategy_settings(exchange)
        if not strategy_settings:
            return _logger.info("No active strategy found, skipping")

        exchange: BaseExchangeAdapter = ExchangeFactory.get_exchange(exchange)
        exchange.load_exchange()
        for strategy in strategy_settings:
            if strategy.id == strategy_id:
                exchange.market = f"{strategy.ticker}"
                trader = TurtleTrader(exchange, strategy)
                trader.close_position()
    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


@cli.command(help='run Turtle trading bot')
@click.option('-exch', '--exchange_id', type=str, default='binance')
def trade(exchange_id):
    _logger.info("\n============== STARTING TRADE SESSION ==============\n")
    try:
        strategy_settings = load_strategy_settings(exchange_id)
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

        exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(exchange_id)
        _logger.info(f"Initialising Turtle trader on {exchange_id}, tickers: {[x.ticker for x in strategy_settings]}")
        exchange_adapter.load_exchange()

        for strategy in strategy_settings_filtered:
            _logger.info(f"\n\n----------- Starting trade - {strategy.ticker}, strategy_id: {strategy.id}-----------")
            exchange_adapter.market = f"{strategy.ticker}"
            trader = TurtleTrader(exchange_adapter, strategy)
            _logger.debug(f"Market info before trading: {exchange_adapter.market_info}")
            trader.trade()

    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}", echo='here')
        sys.exit(1)


@cli.command(help='run LLM trading bot')
@click.option('-exch', '--exchange_id', type=str, default='bybit')
@click.option('-t', '--tickers', type=str, default=None)
def llm_trade(exchange_id, tickers):
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
        else:
            # TickerPicker
            async_exchange_adapter = ExchangeFactory.get_async_exchange(
                exchange_id, sub_account_id=strategy_settings[0].sub_account_id
            )
            llm_ticker_picker = LlmTickerPicker(async_exchange_adapter, strategy_settings)
            strategy_settings_filtered = llm_ticker_picker.llm_pick_tickers()

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

    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier_llm.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


@cli.command(help='run LLM trading bot')
@click.option('-exch', '--exchange_id', type=str, default='binance')
def back_test(exchange_id):
    _logger.info("\n============== BACKTEST ==============\n")
    turtle_back_test(exchange_id)


if __name__ == '__main__':
    init_logging()
    cli()
