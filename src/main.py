import logging
import sys
import traceback

import click
from jnd_utils.log import init_logging
from slack_bot.notifications import SlackNotifier

from config import SLACK_URL, TRADED_TICKERS
from exchange_adapter import ExchangeAdapter
from turtle_trader import TurtleTrader

_logger = logging.getLogger(__name__)
_notifier = SlackNotifier(url=SLACK_URL, username='main')


@click.group(chain=True)
def cli():
    pass


@cli.command()
@click.option('-e', '--exchange', type=str, default='binance')
@click.option('-t', '--ticker', type=str, default='BTC')
def log_pl(exchange, ticker):
    exchange = ExchangeAdapter(exchange)
    exchange.market = f"{ticker}"
    TurtleTrader(exchange).log_total_pl()


@cli.command(help='close position manually')
@click.option('-t', '--ticker', type=str, default='BTC')
def close_position(ticker):
    _logger.info(f"\n============== CLOSING POSITION {ticker} ==============\n")
    try:
        exchange = ExchangeAdapter('binance')
        exchange.load_exchange()
        exchange.market = f"{ticker}"
        trader = TurtleTrader(exchange)
        trader.close_position()
    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


@cli.command(help='run Turtle trading bot')
def trade():
    _logger.info("\n============== STARTING TRADE SESSION ==============\n")
    try:
        _logger.info(f"Initialising Turtle trader, tickers: {TRADED_TICKERS}")
        exchange = ExchangeAdapter('binance')
        exchange.load_exchange()
        for ticker in exchange.exchange_traded_tickers:
            _logger.info(f"\n\n----------- Starting trade - {ticker} -----------")
            exchange.market = f"{ticker}"
            trader = TurtleTrader(exchange)
            _logger.debug(f"Market info before trading: {exchange.market_info}")
            trader.trade()
    except Exception as e:
        _logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        _notifier.error(f"Trading error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    init_logging()
    cli()
