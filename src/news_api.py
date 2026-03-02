import json
import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def get_ticker_news_with_text(ticker_symbol, n=3):
    logger.info(f"Fetching news for ticker: {ticker_symbol}")

    ticker = yf.Ticker(ticker_symbol)

    try:
        news_list = ticker.news
    except Exception as e:
        logger.error(f"Failed to fetch data from Yahoo Finance: {e}")
        return

    if not news_list:
        logger.warning(f"No news found for ticker: {ticker_symbol}")
        return

    # Limit to n latest news
    top_n_news = news_list[:n]
    logger.info(f"Processing {len(top_n_news)} latest articles for {ticker_symbol}")

    news_result = {}

    for i, item in enumerate(top_n_news, 1):
        item_content = item.get('content')
        title = item_content.get('title')
        summary = item_content.get('summary') or item.get('description')

        news_result[i] = {
            'title': title,
            'summary': summary
        }
        logger.debug(f"Article {i} processed: {title[:50]}...")

    # Logování výsledku v čitelném JSON formátu
    logger.info("News fetching completed successfully.")

    return json.dumps(news_result, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Příklad pro PENDLE-USD
    get_ticker_news_with_text("BTC-USD", n=5)
