import logging
import sys
import traceback

import click
from jnd_utils.log import init_logging
from slack_bot.notifications import SlackNotifier

from config import SLACK_URL
from exchange_adapter import BaseExchangeAdapter
from src.exchange_factory import ExchangeFactory
from src.llm_trader import LmmTrader
from src.utils.utils import load_strategy_settings
from turtle_trader import TurtleTrader

_logger = logging.getLogger(__name__)
_notifier = SlackNotifier(url=SLACK_URL, username='main')


@click.group(chain=True)
def cli():
    pass


@cli.command()
@click.option('-exch', '--exchange', type=str, default='binance')
@click.option('-t', '--ticker', type=str, default='BTC')
def log_pl(exchange, ticker):
    exchange: BaseExchangeAdapter = ExchangeFactory.get_exchange(exchange)
    exchange.market = f"{ticker}"
    TurtleTrader(exchange).log_total_pl()


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
        # Using the factory to get the correct exchange adapter
        exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(exchange_id)

        _logger.info(f"Initialising Turtle trader on {exchange_id}, tickers: {[x.ticker for x in strategy_settings]}")
        exchange_adapter.load_exchange()

        for strategy in strategy_settings:
            _logger.info(f"\n\n----------- Starting trade - {strategy.ticker}, strategy_id: {strategy.id}-----------")
            exchange_adapter.market = f"{strategy.ticker}"
            trader = TurtleTrader(exchange_adapter, strategy)
            _logger.debug(f"Market info before trading: {exchange_adapter.market_info}")
            trader.trade()

    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


@cli.command(help='run LLM trading bot')
@click.option('-exch', '--exchange_id', type=str, default='binance')
def llm_trade(exchange_id):
    _logger.info("\n============== STARTING LLM TRADE SESSION ==============\n")
    try:
        strategy_settings = load_strategy_settings(exchange_id, 'llm_trader')
        if not strategy_settings:
            return _logger.info("No active strategy found, skipping")
        # Using the factory to get the correct exchange adapter
        exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(exchange_id)

        _logger.info(f"Initialising LLM trader on {exchange_id}, tickers: {[x.ticker for x in strategy_settings]}")
        exchange_adapter.load_exchange()

        for strategy in strategy_settings:
            _logger.info(f"\n\n----------- Starting trade - {strategy.ticker}, strategy_id: {strategy.id}-----------")
            exchange_adapter.market = f"{strategy.ticker}"
            trader = LmmTrader(exchange_adapter, strategy)
            trader.trade()
            _logger.debug(f"Market info before trading: {exchange_adapter.market_info}")

    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    init_logging()
    cli()
