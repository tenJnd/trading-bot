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
        TurtleTrader(exchange, strategy).log_total_pl()


@cli.command(help='close position manually')
@click.option('-exch', '--exchange', type=str, default='bybit')
@click.option('-si', '--strategy_ids', type=str)
def close_position(exchange, strategy_ids):
    strategy_ids = [int(sid) for sid in strategy_ids.split(',')]
    _logger.info(f"\n============== CLOSING POSITION {strategy_ids} ==============\n")
    agent_map = {
        'turtle_trader': TurtleTrader,
        'shit_trader': ShitTrader
    }
    try:
        strategy_settings = load_strategy_settings(exchange)
        if not strategy_settings:
            return _logger.info("No active strategy found, skipping")

        for strategy in strategy_settings:
            if strategy.id in strategy_ids:
                exchange: BaseExchangeAdapter = ExchangeFactory.get_exchange(
                    exchange,
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
            trader = TurtleTrader(exchange_adapter, strategy)
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
def shit_trade(exchange_id, enter, ticker, timeframe, side, standby_mode):
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
            trader = ShitTrader(exchange_adapter, strategy)
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
@click.option('-exch', '--exchange_id', type=str, default='binance')
def back_test(exchange_id):
    _logger.info("\n============== BACKTEST ==============\n")
    turtle_back_test(exchange_id)


@cli.command(help='run LLM trading bot')
@click.option('-exch', '--exchange_id', type=str, default='binance')
def test_balance(exchange_id):
    # Llm Trader for each picked ticker
    exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange(
        exchange_id, 'subAccount1'
    )
    exchange_adapter.load_exchange()
    exchange_adapter.market = 'BTC'
    balance = exchange_adapter.free_balance
    print(balance)


if __name__ == '__main__':
    init_logging()
    cli()
