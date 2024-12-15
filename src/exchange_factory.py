import logging

from slack_bot.notifications import SlackNotifier

from src.config import SLACK_URL
from src.exchange_adapter import BaseExchangeAdapter, AsyncExchangeAdapter

_notifier = SlackNotifier(url=SLACK_URL, username='Exchange factory')
_logger = logging.getLogger(__name__)


class ExchangeFactory:
    @staticmethod
    def get_exchange(exchange_id: str, sub_account_id: str = None, market: str = None) -> BaseExchangeAdapter:
        _logger.info(f"Creating exchange adapter for {exchange_id}, sub-account: {sub_account_id}")
        if exchange_id in ['binance', 'kucoinfutures', 'bybit', 'mexc']:
            return BaseExchangeAdapter(exchange_id, sub_account_id, market)
        else:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

    @staticmethod
    def get_async_exchange(exchange_id: str, sub_account_id: str = None, market: str = None) -> AsyncExchangeAdapter:
        _logger.info(f"Creating exchange adapter for {exchange_id}, sub-account: {sub_account_id}")
        if exchange_id in ['binance', 'kucoinfutures', 'bybit', 'mexc']:
            return AsyncExchangeAdapter(exchange_id, sub_account_id, market)
        else:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
