import logging

import ccxt
from slack_bot.notifications import SlackNotifier

from src.config import SLACK_URL
from src.exchange_adapter import BaseExchangeAdapter

_notifier = SlackNotifier(url=SLACK_URL, username='Exchange factory')
_logger = logging.getLogger(__name__)


class ExchangeFactory:

    @staticmethod
    def get_exchange(exchange_id: str, market: str = None) -> ccxt.Exchange:
        _logger.info(f"Creating exchange adapter for {exchange_id}")
        if exchange_id in ['binance', 'kucoinfutures', 'bybit']:
            return BaseExchangeAdapter(exchange_id, market)
        else:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
