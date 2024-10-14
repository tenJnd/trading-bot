import json
import math
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import requests

from src.config import TRADING_DATA_DIR, ATR_PERIOD
from src.model import trader_database
from src.model.turtle_model import StrategySettings
from src.schemas.turtle_schema import OrderSchema


def significant_round(num, places):
    """
    Custom rounding to a specific number of significant figures
    starting after the first non-zero digit.

    Parameters:
    num (float): The number to be rounded.
    places (int): The number of significant digits to retain after the first non-zero digit.

    Returns:
    float: The rounded number with the specified significant figures.
    """
    # Get the fractional part
    fraction_part = str(num - int(num))

    # Iterate through the fractional part to find the first non-zero digit
    for i, digit in enumerate(fraction_part[2:], start=2):
        if digit != '0':
            # Concatenate integer part and significant fraction part
            return int(num) + float(fraction_part[:i + places])

    # If no non-zero digit is found, return the original number
    return num


def save_json_to_file(order_data: dict, file_name: str):
    # Create an instance of OrderSchema
    order_schema = OrderSchema()

    # Use OrderSchema instance to dump the order_data into a dictionary
    order_dict = order_schema.dump(order_data)

    # Define the file path
    out_file = os.path.join(TRADING_DATA_DIR, f'{file_name}.json')

    # Write the dictionary as a JSON file
    with open(out_file, "w") as ff:
        json.dump(order_dict, ff, indent=4, ensure_ascii=False)


def get_adjusted_amount(amount, precision):
    if precision == 0:
        return max(1, round(amount))
    else:
        return round(amount, int(precision))  # Ensure precision is an integer


def none_to_default(value, default):
    return default if value is None else value


def dynamic_safe_round(value: float, precision: int = 2) -> float:
    """
    Dynamically rounds the value based on its magnitude:
    - If the value is greater than 1, round to `precision` decimal places (default 1).
    - If the value is less than 1, keep sufficient precision to preserve significant digits,
      but round to 3 decimal places if necessary.

    Args:
        value (float): The value to be rounded.
        precision (int): Number of decimal places to round to for values > 1.

    Returns:
        float: The rounded value.
    """

    # Check for NaN value
    if pd.isna(value):
        return value

    if value == 0:
        return 0.0

    # For values greater than 1, use the standard rounding approach
    if abs(value) >= 1:
        return round(value, precision)

    # For values less than 1, dynamically calculate the decimal places
    # This ensures that the value isn't rounded to a high number like 0.5 or 1
    # We calculate how many leading zeros there are and add that to the precision
    abs_value = abs(value)
    leading_zeros = int(math.floor(abs(math.log10(abs_value))))  # Calculate leading zeros in the decimal part

    # Dynamic precision: higher for smaller values to preserve significant digits
    dynamic_precision = leading_zeros + precision  # +2 ensures we don't over-truncate very small values

    return round(value, dynamic_precision)


def round_series(values, precision: int = 2):
    """
    Applies dynamic_safe_round to a list or a Pandas Series of values.

    Args:
        values (List[float] or pd.Series): A list or Series of float values to be rounded.
        precision (int): Number of decimal places to round to for values > 1.

    Returns:
        List[float]: A list of rounded values.
    """
    return [dynamic_safe_round(val, precision) for val in values]


def calculate_sma(df, period=20, column='C'):
    """Simple Moving Average (SMA)"""
    sma = df[column].rolling(window=period).mean()
    return round_series(sma)


def calculate_ema(df, period=20, column='C'):
    """Exponential Moving Average (EMA)"""
    ema = df[column].ewm(span=period, adjust=False).mean()
    return round_series(ema)


def calculate_rsi(df, period=14, column='C'):
    """Relative Strength Index (RSI)"""
    delta = df[column].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_rounded = round_series(rsi)

    return rsi_rounded


def calculate_macd(df, short_period=12, long_period=26, signal_period=9, column='C'):
    """Moving Average Convergence Divergence (MACD)"""
    short_ema = df[column].ewm(span=short_period, adjust=False).mean()
    long_ema = df[column].ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_line_rounded = round_series(macd_line)
    signal_line_rounded = round_series(signal_line)
    return macd_line_rounded, signal_line_rounded


def calculate_atr(df, period=14):
    """Average True Range (ATR)"""
    high_low = df['H'] - df['L']
    high_close = np.abs(df['H'] - df['C'].shift(1))
    low_close = np.abs(df['L'] - df['C'].shift(1))
    tr = high_low.to_frame().join([high_close, low_close]).max(axis=1)
    return round_series(tr.rolling(window=period).mean())


def calculate_bollinger_bands(df, period=20, column='C', num_std=2):
    """Bollinger Bands"""
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return round_series(sma), round_series(upper_band), round_series(lower_band)


def calculate_stochastic_oscillator(df, period=14, smooth_k=3, smooth_d=3):
    """Stochastic Oscillator"""
    low_min = df['L'].rolling(window=period).min()
    high_max = df['H'].rolling(window=period).max()
    k = 100 * ((df['C'] - low_min) / (high_max - low_min))
    k_smooth = k.rolling(window=smooth_k).mean()
    d = k_smooth.rolling(window=smooth_d).mean()
    return round_series(k_smooth), round_series(d)


def calculate_auto_fibonacci(df, lookback_periods=[5, 10]):
    """
    Calculate Fibonacci retracement levels for multiple periods.

    :param df: DataFrame containing OHLCV data with columns ['O', 'H', 'L', 'C', 'V']
    :param lookback_periods: List of periods to calculate Fibonacci levels for
    :return: Dictionary in the format {lookback_period: {fib_0: value, fib_23.6: value, ...}}
    """
    fib_dict = {}

    for period in lookback_periods:
        # Find the swing high and swing low over the lookback period
        swing_high = df['H'].rolling(window=period).max().iloc[-1]
        swing_low = df['L'].rolling(window=period).min().iloc[-1]

        # Fibonacci retracement levels
        fib_levels = [0, 23.6, 38.2, 50.0, 61.8, 100]

        # Calculate Fibonacci retracement levels and store them in a dictionary
        fib_values = {
            f'fib_{level}': dynamic_safe_round(swing_high - (swing_high - swing_low) * (level / 100)) for level in
            fib_levels
        }

        # Add the swing high and swing low to the dictionary
        fib_values['swing_high'] = swing_high
        fib_values['swing_low'] = swing_low

        # Store the values in the final dictionary for the given period
        fib_dict[period] = fib_values

    return fib_dict


def calculate_pivot_points(df, lookback_periods=[5, 10]):
    """
    Calculate Pivot Points, Support, and Resistance levels for multiple periods.

    :param df: DataFrame containing OHLCV data with columns ['O', 'H', 'L', 'C', 'V']
    :param lookback_periods: List of periods to calculate pivot points for
    :return: Dictionary in the format {lookback_period: {pivot: value, s1: value, r1: value, ...}}
    """
    pivot_dict = {}

    for period in lookback_periods:
        # Calculate the recent high, low, and close over the lookback period
        high = df['H'].rolling(window=period).max().iloc[-1]
        low = df['L'].rolling(window=period).min().iloc[-1]
        close = df['C'].rolling(window=period).apply(lambda x: x[-1], raw=True).iloc[-1]

        # Calculate pivot points
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        r1 = (2 * pivot) - low
        s2 = pivot - (high - low)
        r2 = pivot + (high - low)

        # Store the pivot points in a dictionary
        pivot_values = {
            'pivot': dynamic_safe_round(pivot),
            's1': dynamic_safe_round(s1),
            'r1': dynamic_safe_round(r1),
            's2': dynamic_safe_round(s2),
            'r2': dynamic_safe_round(r2),
            'high': dynamic_safe_round(high),
            'low': dynamic_safe_round(low),
            'close': dynamic_safe_round(close)
        }

        # Store the values in the final dictionary for the given period
        pivot_dict[period] = pivot_values

    return pivot_dict


def calculate_atr(df, period=ATR_PERIOD, long_period=50):
    """
    Calculate the Average True Range (ATR) for given OHLCV DataFrame.

    Parameters:
    - df: pandas DataFrame with columns 'H', 'L', and 'C'.
    - period: the period over which to calculate the ATR.

    Returns:
    - A pandas Series representing the ATR.
    """
    # Calculate true ranges
    df['high_low'] = df['H'] - df['L']
    df['high_prev_close'] = abs(df['H'] - df['C'].shift(1))
    df['low_prev_close'] = abs(df['L'] - df['C'].shift(1))

    # Find the max of the true ranges
    df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)

    # Calculate the ATR
    df['atr_20'] = round_series(df['true_range'].rolling(window=period, min_periods=1).mean())
    df['atr_50'] = round_series(df['true_range'].rolling(window=long_period, min_periods=1).mean())

    # Clean up the DataFrame by removing the intermediate columns
    df.drop(['high_low', 'high_prev_close', 'low_prev_close', 'true_range'], axis=1, inplace=True)

    return df


def calculate_adx(df, n_periods=14):
    """
    Calculate the Average Directional Index (ADX) to measure trend strength.
    Returns the ADX as a Series without modifying the DataFrame.

    :param df: DataFrame with OHLC data
    :param n_periods: Number of periods for ADX calculation
    :return: Series with ADX values
    """
    # Calculate True Range (TR)
    tr1 = df['H'] - df['L']
    tr2 = abs(df['H'] - df['C'].shift(1))
    tr3 = abs(df['L'] - df['C'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement (DM)
    up_move = df['H'] - df['H'].shift(1)
    down_move = df['L'].shift(1) - df['L']
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    # Calculate the smoothed DM and TR values
    plus_di = 100 * (plus_dm.rolling(window=n_periods).mean() / true_range.rolling(window=n_periods).mean())
    minus_di = 100 * (minus_dm.rolling(window=n_periods).mean() / true_range.rolling(window=n_periods).mean())

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=n_periods).mean()

    return round_series(adx)


def fetch_binance_open_interest(symbol, period='4h', limit=50, start_time=None, end_time=None):
    """
    Fetch historical open interest data from Binance Futures API.

    :param symbol: The market symbol (e.g., 'BTCUSDT')
    :param period: The time period (e.g., '5m', '15m', '1h', '4h', '1d')
    :param limit: Number of records to fetch (max 500)
    :param start_time: Optional start time in milliseconds (Unix timestamp)
    :param end_time: Optional end time in milliseconds (Unix timestamp)
    :return: JSON response containing open interest data or None if the request fails
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"

    # Define query parameters
    params = {
        'symbol': symbol,
        'period': period,
        'limit': limit
    }

    # Add optional start and end time if provided
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time

    try:
        # Make the GET request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Return the parsed JSON response
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Status code: {response.status_code}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")

    # Return None in case of failure
    return None


@dataclass
class StrategySettingsModel:
    id: int
    exchange_id: str
    ticker: str
    timeframe: str
    buffer_days: int
    stop_loss_atr_multipl: int
    pyramid_entry_atr_multipl: float
    aggressive_pyramid_entry_multipl: float
    aggressive_price_atr_ratio: float
    pyramid_entry_limit: int

    @classmethod
    def from_orm(cls, orm_obj):
        # Use helper function to replace None with default
        return cls(
            id=orm_obj.id,
            exchange_id=orm_obj.exchange_id,
            ticker=orm_obj.ticker,
            timeframe=orm_obj.timeframe,
            buffer_days=none_to_default(orm_obj.buffer_days, 60),
            stop_loss_atr_multipl=none_to_default(orm_obj.stop_loss_atr_multipl, 2),
            pyramid_entry_atr_multipl=none_to_default(orm_obj.pyramid_entry_atr_multipl, 1),
            aggressive_pyramid_entry_multipl=none_to_default(orm_obj.aggressive_pyramid_entry_multipl, 0.5),
            aggressive_price_atr_ratio=none_to_default(orm_obj.aggressive_price_atr_ratio, 0.02),
            pyramid_entry_limit=none_to_default(orm_obj.pyramid_entry_limit, 4)
        )


def load_strategy_settings(exchange_id, agent_id: str = 'turtle_trader') -> List[StrategySettingsModel]:
    with trader_database.session_manager() as session:
        # Querying only the necessary columns
        settings = session.query(
            StrategySettings.id,
            StrategySettings.exchange_id,
            StrategySettings.ticker,
            StrategySettings.timeframe,
            StrategySettings.buffer_days,
            StrategySettings.stop_loss_atr_multipl,
            StrategySettings.pyramid_entry_atr_multipl,
            StrategySettings.aggressive_pyramid_entry_multipl,
            StrategySettings.aggressive_price_atr_ratio,
            StrategySettings.pyramid_entry_limit,
        ).filter(
            StrategySettings.exchange_id == exchange_id,
            StrategySettings.active == True,
            StrategySettings.agent_id == agent_id
        ).order_by(
            StrategySettings.timestamp_created
        ).all()

    # Convert the SQLAlchemy result rows to StrategySettingsModel using from_orm
    result_objects = [
        StrategySettingsModel.from_orm(row)
        for row in settings
    ]

    return result_objects
