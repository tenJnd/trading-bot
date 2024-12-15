import asyncio
import logging
from typing import List

import pandas as pd
from slack_bot.notifications import SlackNotifier

from config import SLACK_URL, LLM_TRADER_SLACK_URL
from src.llm_trader import LlmTrader
from src.prompts import ticker_picker_prompt
from src.utils.utils import StrategySettingsModel, calculate_indicators_for_llm_trader, \
    calculate_auto_fibonacci

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
        self.open_strategies = []

        self.llm_input_data = self.create_llm_input_dict()

    async def async_get_tickers_data(self):
        _logger.info("async - fetching ohlc and open positions")

        await self._exchange.load_exchange()

        op = await self._exchange.get_open_positions()
        tic = await self._exchange.async_fetch_ohlc(self.tickers_input)

        await self._exchange.close()
        return tic, op

    def create_llm_input_dict(self):
        """
        Synchronous wrapper for the asynchronous trading bot function.
        """
        _logger.info("Generating llm input")
        tickers_data, open_positions = asyncio.run(self.async_get_tickers_data())
        open_tickers = [op['symbol'].split('/')[0] for op in open_positions]
        _logger.info(f'Opened positions {open_tickers}')
        self.open_strategies = [strategy for strategy in self.strategies if strategy.ticker in open_tickers]

        ticker_list = []
        fib_list = []

        for tic in tickers_data[0]:
            tic = calculate_indicators_for_llm_trader(tic)
            tic_fib = calculate_auto_fibonacci(tic, lookback_periods=[50])

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

        return str({
            'price_data': all_df_csv,
            'auto_fib_data': all_fib_df_csv
        })

    @staticmethod
    def generate_functions():
        return [
            {
                "name": "trading_decision",
                "description": "Pick tickers with the best trading potential and provide their scores.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "object",  # This allows for a dictionary structure
                            "additionalProperties": {
                                "type": "integer",  # Scores are integers
                                "description": "Score for the ticker based on trading potential (1-100)."
                            },
                            "description": "A dictionary of tickers and their scores."
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why the selected tickers were chosen."
                        },
                    },
                    "required": ["action", "rationale"]  # Both fields are required
                }
            }
        ]

    def pick_tickers(self):
        _logger.info('Picking tradeable tickers...')
        agent_action = self.call_agent()

        sorted_desc_values = dict(
            sorted(agent_action.action.items(), key=lambda item: item[1], reverse=True)[:2]
        )
        _logger.info(f"Ticker picker selection: {sorted_desc_values}\n"
                     f"rationale: {agent_action.rationale}")

        possible_strategies = [
            strategy for strategy in self.strategies
            if strategy.ticker in sorted_desc_values.keys()
        ]
        strategies_to_trade = possible_strategies + self.open_strategies
        return strategies_to_trade
