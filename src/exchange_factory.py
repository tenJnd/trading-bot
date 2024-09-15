import logging

import ccxt
from slack_bot.notifications import SlackNotifier

from config import SLACK_URL
from src.exchange_adapter import BaseExchangeAdapter, MexcExchange

_notifier = SlackNotifier(url=SLACK_URL, username='Exchange factory')
_logger = logging.getLogger(__name__)


class ExchangeFactory:

    @staticmethod
    def get_exchange(exchange_id: str, market: str = None) -> ccxt.Exchange:
        _logger.info(f"Creating exchange adapter for {exchange_id}")
        if exchange_id in ['binance', 'kucoinfutures']:
            return BaseExchangeAdapter(exchange_id, market)
        elif exchange_id == 'mexc':
            return MexcExchange(exchange_id, market)
        else:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
