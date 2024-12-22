import asyncio
import logging
from typing import List

import pandas as pd
from slack_bot.notifications import SlackNotifier

from config import SLACK_URL, LLM_TRADER_SLACK_URL
from src.llm_trader import LlmTrader
from src.prompts import ticker_picker_prompt
from src.utils.utils import StrategySettingsModel, calculate_indicators_for_llm_trader, \
    calculate_auto_fibonacci, turtle_trading_signals_adjusted

_logger = logging.getLogger(__name__)
_notifier = SlackNotifier(url=SLACK_URL, username='turtle_trader')
_notifier_llm = SlackNotifier(url=LLM_TRADER_SLACK_URL, username='llm_trader')


class LlmTickerPicker(LlmTrader):
    agent_name = 'ticker_picker'
    system_prompt = ticker_picker_prompt

    def __init__(self, exchange, strategies: List[StrategySettingsModel]):
        _logger.info("Initiating TickerPicker...")
        super().__init__(exchange=exchange, load_data=False)
        self.strategies = strategies
        self.tickers_input = [f"{strategy.ticker}/USDT:USDT" for strategy in strategies]
        self.open_positions_tickers = []
        self.open_orders_tickers = []

        self.create_llm_input_dict()

    async def async_fetch_data(self):
        _logger.info("async - fetching ohlc and open positions")
        timeframe = self.strategies[0].timeframe
        buffer_days = self.strategies[0].buffer_days

        await self._exchange.load_exchange()
        await self._exchange.fetch_balance()

        op = await self._exchange.get_open_positions_all()
        oo = await self._exchange.get_opened_orders_all(self.tickers_input)
        self.open_positions_tickers = [op['symbol'].split('/')[0] for op in op]
        self.open_orders_tickers = [op['symbol'].split('/')[0] for op in oo]

        tic = await self._exchange.async_fetch_ohlc(self.tickers_input, days=buffer_days, timeframe=timeframe)

        await self._exchange.close()
        return tic

    def create_llm_input_dict(self):
        """
        Synchronous wrapper for the asynchronous trading bot function.
        """
        _logger.info("Generating llm input")
        tickers_data = asyncio.run(self.async_fetch_data())
        _logger.info(f'Opened positions {self.open_positions_tickers}')

        ticker_list = []
        fib_list = []

        for tic in tickers_data[0]:
            tic = calculate_indicators_for_llm_trader(tic)
            tic_fib = calculate_auto_fibonacci(tic, lookback_periods=[20])

            tic_last = tic.iloc[-1]
            ticker_list.append(tic_last)

            ticker_symbol = tic_last.ticker
            close_price = tic_last.C

            for fib in tic_fib:
                fib['ticker'] = ticker_symbol
                fib['C'] = close_price
                fib_list.append(fib)

        all_df = pd.DataFrame(ticker_list)
        all_fib_df = pd.DataFrame(fib_list)

        all_df_csv = all_df.to_csv(index=False)
        all_fib_df_csv = all_fib_df.to_csv(index=False)

        self.llm_input_data = str({
            'price_data': all_df_csv,
            'auto_fib_data': all_fib_df_csv
        })

    @staticmethod
    def generate_functions():
        return [
            {
                "name": "trading_decision",
                "description": "Analyzes crypto tickers and returns a decision on their trading potential.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string", "description": "The symbol of the crypto ticker."},
                                    "score": {
                                        "type": "integer",
                                        "description": "Score for the ticker based on trading potential (1-100)."
                                    }
                                },
                                "required": ["ticker", "score"]
                            },
                            "description": "A list of tickers with their associated scores."
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why the selected tickers are promising, based on data analysis."
                        }
                    },
                    "required": ["data", "rationale"]
                }
            }
        ]

    def llm_pick_tickers(self):
        _logger.info('Picking tradable tickers...')
        free_balance = self._exchange.free_balance
        total_balance = self._exchange.total_balance
        free_cap_ratio = free_balance / total_balance

        high_score_tickers_symbols = []
        if free_cap_ratio >= 0.3:
            agent_action = self.call_agent()
            agent_action.action = 'ticker_pick'

            # Extract ticker and score from the list of dictionaries and filter them based on score > 85
            high_score_tickers = [ticker for ticker in agent_action.data if ticker['score'] > 80]

            msg = (f"Ticker picker selection based on high score: {high_score_tickers}\n"
                   f"rationale: {agent_action.rationale}")
            _logger.info(msg)
            _notifier_llm.info(msg)

            # Extract tickers for easier comparison
            high_score_tickers_symbols = [ticker['ticker'] for ticker in high_score_tickers]
        else:
            _logger.info(f"Free capital ratio: ({round(free_cap_ratio * 100, 1)}%),"
                         f" we'll process only open positions")

        # Combine unique symbols from open strategies and high-score tickers
        possible_symbols = set(
            self.open_positions_tickers +
            self.open_orders_tickers +
            high_score_tickers_symbols
        )

        # Filter strategies that are applicable to the high score tickers
        possible_strategies = [
            strategy for strategy in self.strategies
            if strategy.ticker in possible_symbols
        ]
        return possible_strategies


class TickerPicker:
    agent_name = 'ticker_picker'
    system_prompt = ticker_picker_prompt

    def __init__(self, exchange, strategies: List[StrategySettingsModel]):
        _logger.info("Initiating TickerPicker...")
        self._exchange = exchange
        self.strategies = strategies
        self.tickers_input = [f"{strategy.ticker}/USDT:USDT" for strategy in strategies]
        self.open_positions_tickers = []

    async def async_get_tickers_data(self):
        _logger.info("async - fetching ohlc and open positions")
        timeframe = self.strategies[0].timeframe
        buffer_days = self.strategies[0].buffer_days

        await self._exchange.load_exchange()

        op = await self._exchange.get_open_positions_all()
        tic = await self._exchange.async_fetch_ohlc(self.tickers_input, days=buffer_days, timeframe=timeframe)

        await self._exchange.close()
        return tic, op

    def pick_tickers(self):

        tickers_data, open_positions = asyncio.run(self.async_get_tickers_data())
        self.open_positions_tickers = [op['symbol'].split('/')[0] for op in open_positions]
        _logger.info(f'Opened positions {self.open_positions_tickers}')

        ticker_list = []
        for tic in tickers_data[0]:
            tic = turtle_trading_signals_adjusted(tic)
            tic_last = tic.iloc[-1]
            if tic_last.long_entry or tic_last.short_entry:
                ticker_list.append(tic_last.ticker)

        result_list = set(self.open_positions_tickers + ticker_list)
        _logger.info(f"Filtered tickers: {result_list}")

        # Filter strategies that are applicable to the high score tickers
        possible_strategies = [
            strategy for strategy in self.strategies
            if strategy.ticker in result_list
        ]
        return possible_strategies
