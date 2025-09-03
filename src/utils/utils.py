import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List

import numpy as np
import pandas as pd

from src.config import TRADING_DATA_DIR, TURTLE_ENTRY_DAYS, TURTLE_EXIT_DAYS
from src.model import trader_database
from src.model.turtle_model import StrategySettings, BalanceReport
from src.schemas.turtle_schema import OrderSchema

_logger = logging.getLogger(__name__)


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
    amount = Decimal(str(amount))
    precision = Decimal(str(precision))

    if precision <= 0:
        # If precision is invalid, default to nearest integer
        return max(1, round(amount))

    # Adjust the amount to the nearest multiple of precision
    return float((amount / precision).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * precision)


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
    dynamic_precision = leading_zeros + precision

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


def turtle_trading_signals_adjusted(df):
    """
    Identify Turtle Trading entry and exit signals for both long and short positions, adjusting for early rows.

    Parameters:
    - df: pandas DataFrame with at least 'High' and 'Low' columns.

    Adds columns to df:
    - 'long_entry': Signal for entering a long position.
    - 'long_exit': Signal for exiting a long position.
    - 'short_entry': Signal for entering a short position.
    - 'short_exit': Signal for exiting a short position.
    """
    # df['datetime'] = pd.to_datetime(df['timeframe'], unit='ms')
    # Calculate rolling max/min for the required windows with min_periods=1
    df['high_20'] = df['H'].rolling(window=TURTLE_ENTRY_DAYS, min_periods=1).max()
    df['low_20'] = df['L'].rolling(window=TURTLE_ENTRY_DAYS, min_periods=1).min()
    df['high_10'] = df['H'].rolling(window=TURTLE_EXIT_DAYS, min_periods=1).max()
    df['low_10'] = df['L'].rolling(window=TURTLE_EXIT_DAYS, min_periods=1).min()

    # Entry signals
    df['long_entry'] = df['H'] > df['high_20'].shift(1)
    df['short_entry'] = df['L'] < df['low_20'].shift(1)

    # Exit signals
    df['long_exit'] = df['L'] < df['low_10'].shift(1)
    df['short_exit'] = df['H'] > df['high_10'].shift(1)

    df['diff'] = df.C.diff()

    df = df.drop(['high_20', 'high_10', 'low_20', 'low_10'], axis=1)
    return df


def calculate_sma(df, period=20, column='C'):
    """Simple Moving Average (SMA)"""
    sma = df[column].rolling(window=period).mean()
    return round_series(sma, precision=4)


def calculate_ema(df, period=20, column='C'):
    """Exponential Moving Average (EMA)"""
    ema = df[column].ewm(span=period, adjust=False).mean()
    return round_series(ema, precision=4)


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


def calculate_obv(df):
    """
    Calculate the On-Balance Volume (OBV) indicator.

    :param df: DataFrame containing OHLCV data with columns ['O', 'H', 'L', 'C', 'V']
    :return: Series containing the OBV values
    """
    # Calculate price difference
    price_change = df['C'].diff()
    # Calculate OBV based on the price change
    obv = np.where(price_change > 0, df['V'],
                   np.where(price_change < 0, -df['V'], 0)).cumsum()
    return pd.Series(obv, index=df.index)


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


def calculate_auto_fibonacci_simple_period(df, lookback_periods=[5, 10]):
    """
    Calculate Fibonacci retracement or extension levels for multiple periods.

    :param df: DataFrame containing OHLCV data with columns ['O', 'H', 'L', 'C', 'V']
    :param lookback_periods: List of periods to calculate Fibonacci levels for
    :return: List of dictionaries in the format:
             {lookback_period: {fib_0: value, fib_23.6: value, ..., type: 'retracement' or 'extension'}}
    """
    fib_dict = []

    for period in lookback_periods:
        # Find the swing high and swing low over the lookback period
        swing_high = df['H'].rolling(window=period).max().iloc[-1]
        swing_low = df['L'].rolling(window=period).min().iloc[-1]

        # Determine if it is a retracement or extension based on the current close price
        current_close = df['C'].iloc[-1]
        if swing_high >= current_close >= swing_low:
            fib_type = "retracement"
            fib_levels = [0, 23.6, 38.2, 50.0, 61.8, 100]
            fib_values = {
                f'fib_{level}': swing_low + (swing_high - swing_low) * (level / 100)
                for level in fib_levels
            }
        elif current_close >= swing_high:
            fib_type = "extension"
            fib_levels = [100, 123.6, 138.2, 150.0, 161.8, 200]
            fib_values = {
                f'fib_{level}': swing_high + (swing_high - swing_low) * ((level - 100) / 100)
                for level in fib_levels
            }
        else:
            raise ValueError(
                "Invalid state for Fibonacci calculation. Close price does not indicate retracement or extension.")

        # Add the swing high, swing low, and additional metadata to the dictionary
        fib_values['swing_high'] = swing_high
        fib_values['swing_low'] = swing_low
        fib_values['fib_period'] = period
        fib_values['type'] = fib_type

        fib_dict.append(fib_values)

    return fib_dict


def calculate_pivot_points(df, lookback_periods=[5, 10]):
    """
    Calculate Pivot Points, Support, and Resistance levels for multiple periods.

    :param df: DataFrame containing OHLCV data with columns ['O', 'H', 'L', 'C', 'V']
    :param lookback_periods: List of periods to calculate pivot points for
    :return: Dictionary in the format {lookback_period: {pivot: value, s1: value, r1: value, ...}}
    """
    pivot_dict = []

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
            'pivot_points_period': period,
            'pivot': dynamic_safe_round(pivot),
            's1': dynamic_safe_round(s1),
            'r1': dynamic_safe_round(r1),
            's2': dynamic_safe_round(s2),
            'r2': dynamic_safe_round(r2),
            'high': dynamic_safe_round(high),
            'low': dynamic_safe_round(low),
            'close': dynamic_safe_round(close)
        }

        pivot_dict.append(pivot_values)

    return pivot_dict


def calculate_closest_fvg_zones(df, current_price, threshold_percentage=2.0):
    """
    Detects and returns the closest significant bullish and bearish Fair Value Gaps (FVGs) to the current price
    in a DataFrame containing OHLCV data, where significance is defined by a percentage threshold.

    :param df: DataFrame containing OHLCV data with columns ['O', 'H', 'L', 'C', 'V']
    :param current_price: The current price around which the closest FVGs are to be identified.
    :param threshold_percentage: The minimum percentage difference required for an FVG to be considered significant.
    :return: Dictionary with closest significant bullish and bearish FVG details.
    """
    df = df.copy()
    df = df[:-1]
    df = df.tail(100)
    # Initialize list to store FVG zones and variables to track the closest FVGs
    closest_bullish = None
    closest_bearish = None
    min_bullish_distance = float('inf')
    min_bearish_distance = float('inf')

    # Iterate over the DataFrame
    for i in range(2, len(df)):
        prev_high = df['H'].iloc[i - 2]
        prev_low = df['L'].iloc[i - 2]
        curr_high = df['H'].iloc[i]
        curr_low = df['L'].iloc[i]

        # Calculate the percentage difference for bullish and bearish FVG candidates
        bullish_percentage_diff = ((curr_low - prev_high) / prev_high) * 100
        bearish_percentage_diff = ((prev_low - curr_high) / curr_high) * 100

        # Detect Bullish FVG with threshold
        if bullish_percentage_diff > threshold_percentage:
            fvg = {
                "type": "bullish",
                "upper_bound": curr_low,
                "lower_bound": prev_high,
                "start_index": i - 2,
                "end_index": i
            }
            distance = abs(current_price - curr_low)
            if distance < min_bullish_distance:
                min_bullish_distance = distance
                closest_bullish = fvg

        # Detect Bearish FVG with threshold
        if bearish_percentage_diff > threshold_percentage:
            fvg = {
                "type": "bearish",
                "upper_bound": prev_low,
                "lower_bound": curr_high,
                "start_index": i - 2,
                "end_index": i
            }
            distance = abs(current_price - prev_low)
            if distance < min_bearish_distance:
                min_bearish_distance = distance
                closest_bearish = fvg

    # Prepare the result dictionary
    result = {
        "bullish": closest_bullish,
        "bearish": closest_bearish
    }

    return result


def calculate_regression_channels(df, length=100, source_column='C', upper_dev=2.0, lower_dev=2.0):
    _logger.info('Calculating regression channel...')
    if len(df) < length:
        raise ValueError("DataFrame is shorter than the specified length for regression calculation.")

    # Prepare lists to store the values
    regression_vals = [np.nan] * len(df)
    upper_channel_vals = [np.nan] * len(df)
    lower_channel_vals = [np.nan] * len(df)

    # Time index for regression calculation for each window
    time_index = np.arange(length)

    for start_idx in range(len(df) - length + 1):
        end_idx = start_idx + length

        # Select the source data
        source_data = df[source_column].iloc[start_idx:end_idx]

        # Calculate the coefficients of the linear regression
        slope, intercept = np.polyfit(time_index, source_data, 1)

        # Calculate the regression values and channels
        regression_values = slope * time_index + intercept
        residuals = source_data - regression_values
        standard_deviation = np.std(residuals)
        upper_channel = regression_values + upper_dev * standard_deviation
        lower_channel = regression_values - lower_dev * standard_deviation

        # Store calculated values in lists
        regression_vals[start_idx:end_idx] = regression_values
        upper_channel_vals[start_idx:end_idx] = upper_channel
        lower_channel_vals[start_idx:end_idx] = lower_channel

    # Assign lists to DataFrame columns
    df['regression'] = round_series(regression_vals, 3)
    df['regression_upper_channel'] = round_series(upper_channel_vals, 3)
    df['regression_lower_channel'] = round_series(lower_channel_vals, 3)

    return df


def calculate_atr(df, period):
    # A simple ATR calculation for testing purposes
    high_low = df['H'] - df['L']
    high_close = np.abs(df['H'] - df['C'].shift())
    low_close = np.abs(df['L'] - df['C'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


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


def format_large_number(num):
    """
    Convert a large number into a shorter format (e.g., 1K, 1M, 1B).

    :param num: The number to be formatted
    :return: Formatted string representing the shortened version of the number
    """
    if abs(num) >= 1_000_000_000:
        return f'{num / 1_000_000_000:.3f}B'
    elif abs(num) >= 1_000_000:
        return f'{num / 1_000_000:.2f}M'
    elif abs(num) >= 1_000:
        return f'{num / 1_000:.1f}K'
    else:
        return str(num)


def shorten_large_numbers(df, column_name):
    """
    Apply the format_large_number function to a specific column of the DataFrame.

    :param df: The DataFrame containing the column to be formatted
    :param column_name: The name of the column to be formatted
    :return: DataFrame with the specified column formatted
    """
    # Apply the format_large_number function to the specified column
    df[column_name] = df[column_name].apply(format_large_number)

    return df


def add_live_bar_time_features(df):
    """
    Assumes `time_col` holds epoch ms integers.
    Adds columns:
      is_live_bar, bar_elapsed_s, bar_progress, volume_rate,
      est_fullbar_volume, vol_rate_vs_avg, datetime (UTC)
    """
    if len(df) == 0:
        return df

    # 2) Mark live bar (last row)
    df['is_live_bar'] = False
    df.loc[df.index[-1], 'is_live_bar'] = True

    # 3) Infer timeframe seconds (robust median of diffs)
    time_deltas = df['datetime'].diff().dropna()
    if not time_deltas.empty:
        timeframe_seconds = int(time_deltas.dt.total_seconds().median())
    else:
        # Fallback if only one row; we can’t infer the frame
        timeframe_seconds = 0

    # Initialize outputs
    df['bar_elapsed_s'] = 0.0
    df['bar_progress'] = 0.0
    # df['volume_rate'] = np.nan
    # df['est_fullbar_volume'] = np.nan
    # df['vol_rate_vs_avg'] = np.nan

    # 4) Compute live-bar features only if we have a valid frame
    if df.iloc[-1]['is_live_bar'] and timeframe_seconds > 0:
        last_open_time = df.iloc[-1]['datetime']  # tz-aware (UTC)
        now_utc = pd.Timestamp.now(tz='UTC')  # tz-aware (UTC)

        elapsed = (now_utc - last_open_time).total_seconds()
        # Clamp to [0, timeframe_seconds]
        elapsed = max(0.0, min(elapsed, float(timeframe_seconds)))

        prog = max(1e-6, min(elapsed / float(timeframe_seconds), 0.999))

        df.loc[df.index[-1], 'bar_elapsed_s'] = round(elapsed, 0)
        df.loc[df.index[-1], 'bar_progress'] = round(prog, 3)

        # V = float(df.iloc[-1]['V'])
        # df.loc[df.index[-1], 'volume_rate'] = V / max(elapsed, 1e-6)
        # df.loc[df.index[-1], 'est_fullbar_volume'] = V / prog
        #
        # if 'volume_sma_20' in df.columns and df.iloc[-1]['volume_sma_20'] not in (0, np.nan):
        #     df.loc[df.index[-1], 'vol_rate_vs_avg'] = (
        #         df.iloc[-1]['est_fullbar_volume'] / df.iloc[-1]['volume_sma_20']
        #     )

    return df


def calculate_indicators_for_llm_trader(df):
    # df['volume_sma_10'] = calculate_sma(df, period=10, column='V')
    df['volume_sma_20'] = calculate_sma(df, period=20, column='V')
    df['atr_20'] = calculate_atr(df, period=20)
    # df['atr_50'] = calculate_atr(df, period=50)
    df['sma_10'] = calculate_sma(df, period=10)
    df['sma_20'] = calculate_sma(df, period=20)
    df['sma_50'] = calculate_sma(df, period=50)
    df['sma_100'] = calculate_sma(df, period=100)
    # df['vol_sma_20'] = calculate_sma(df, period=20, column='V')
    # df['vol_sma_50'] = calculate_sma(df, period=50, column='V')
    # df['rsi_14'] = calculate_rsi(df, period=14)
    # df['rsi_sma_14'] = calculate_sma(df, period=14, column='rsi_14')
    df['macd_12_26'], df['macd_signal_9'] = calculate_macd(df)
    df['bollinger_band_middle_20'], df['bollinger_band_upper_20'], df[
        'bollinger_band_lower_20'] = calculate_bollinger_bands(df)
    # df['stochastic_k_14_s3'], df['stochastic_d_14_s3'] = calculate_stochastic_oscillator(df)
    df['adx_20'] = calculate_adx(df, n_periods=20)
    # df['obv'] = round_series(calculate_obv(df), 0)
    # df['obv_sma_20'] = round_series(calculate_sma(df, period=20, column='obv'), 0)
    df = add_live_bar_time_features(df)

    return df


def calculate_indicators_for_llm_entry_validator(df):
    df = turtle_trading_signals_adjusted(df)
    df = calculate_indicators_for_llm_trader(df)
    df = df.drop(['short_exit', 'long_exit'], axis=1)
    return df


def calculate_indicators_for_nn_30m(df):
    df['volume_sma_7'] = calculate_sma(df, period=7, column='V')
    df['volume_sma_14'] = calculate_sma(df, period=14, column='V')
    df['atr_20'] = calculate_atr(df, period=24)
    df['atr_50'] = calculate_atr(df, period=48)
    df['sma_7'] = calculate_sma(df, period=7)
    df['sma_14'] = calculate_sma(df, period=14)
    df['sma_28'] = calculate_sma(df, period=28)
    df['sma_48'] = calculate_sma(df, period=48)
    df['rsi_14'] = calculate_rsi(df, period=14)
    df['rsi_sma_14'] = calculate_sma(df, period=14, column='rsi_14')
    df['macd_12_26'], df['macd_signal_9'] = calculate_macd(df)
    df['bollinger_band_middle_20'], df['bollinger_band_upper_20'], df[
        'bollinger_band_lower_20'] = calculate_bollinger_bands(df)
    df['adx_24'] = calculate_adx(df, n_periods=24)
    df = calculate_regression_channels(df, length=100)
    df = calculate_fib_levels_rolling(df, depth=100)
    return df


def calculate_indicators_for_nn_4h(df):
    df['volume_sma_10'] = calculate_sma(df, period=10, column='V')
    df['volume_sma_50'] = calculate_sma(df, period=50, column='V')
    df['atr_20'] = calculate_atr(df, period=20)
    df['atr_50'] = calculate_atr(df, period=50)
    df['sma_20'] = calculate_sma(df, period=20)
    df['sma_50'] = calculate_sma(df, period=50)
    df['sma_100'] = calculate_sma(df, period=100)
    df['rsi_14'] = calculate_rsi(df, period=14)
    df['rsi_sma_14'] = calculate_sma(df, period=14, column='rsi_14')
    df['macd_12_26'], df['macd_signal_9'] = calculate_macd(df)
    df['bollinger_band_middle_20'], df['bollinger_band_upper_20'], df[
        'bollinger_band_lower_20'] = calculate_bollinger_bands(df)
    df['adx_20'] = calculate_adx(df, n_periods=20)
    df = calculate_regression_channels(df, length=50)
    df = calculate_fib_levels_rolling(df, depth=50)
    return df


def time_ago_string(timestamp_created: datetime) -> str:
    now = datetime.now(timezone.utc)
    time_diff = now - timestamp_created

    # Convert time difference to seconds
    seconds = time_diff.total_seconds()

    # Determine the time difference and return the appropriate string
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = int(seconds // 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"


def find_pivots(df, depth, deviation):
    """Identifies the last pivot highs and lows in a DataFrame."""
    atr = calculate_atr(df, period=depth)
    deviation_threshold = atr * (deviation / 100.0)

    high_deviation = df['H'] > (df['H'].shift(1) + deviation_threshold)
    low_deviation = df['L'] < (df['L'].shift(1) - deviation_threshold)

    pivot_highs = (df['H'].rolling(window=depth, center=True).max() == df['H']) & high_deviation
    pivot_lows = (df['L'].rolling(window=depth, center=True).min() == df['L']) & low_deviation

    last_pivot_high_index = pivot_highs[pivot_highs].last_valid_index()
    last_pivot_low_index = pivot_lows[pivot_lows].last_valid_index()

    if last_pivot_high_index is not None and last_pivot_low_index is not None:
        last_pivot_high = df.loc[last_pivot_high_index, 'H']
        last_pivot_low = df.loc[last_pivot_low_index, 'L']

        pivot_high = (last_pivot_high, last_pivot_high_index)
        pivot_low = (last_pivot_low, last_pivot_low_index)

        return pivot_high, pivot_low
    else:
        return None


def calculate_fib_levels_pivots(df, depth=20, deviation=3):
    result = find_pivots(df, depth, deviation)
    if result:
        pivot_high, pivot_low = result
        high_price, high_index = pivot_high
        low_price, low_index = pivot_low

        # Determine trend to decide start and end price for Fib levels
        if high_index < low_index:
            start_price = low_price
            end_price = high_price
        else:
            start_price = high_price
            end_price = low_price

        levels = [-0.618, -0.236, 0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.272, 1.618]
        fib_levels = {f"fib_{abs(level):.3f}": start_price + (end_price - start_price) * level for level in levels}
        return fib_levels
    return None  # Return None if the pivots could not be determined


def calculate_fib_levels_rolling(df, depth=20):
    _logger.info("Calculating fib levels rolling...")
    last_pivot_high = None
    last_pivot_low = None

    for index, row in df.iterrows():
        current_high = row['H']
        current_low = row['L']

        if last_pivot_high is None or current_high > last_pivot_high[0]:
            last_pivot_high = (current_high, index)
        if last_pivot_low is None or current_low < last_pivot_low[0]:
            last_pivot_low = (current_low, index)

        if index >= depth + 1 and last_pivot_high and last_pivot_low:
            sub_df = df.loc[:index]
            fib_levels = calculate_fib_levels_pivots(sub_df)  # TODO: from 0-1.618, (-) not included in reinforce
            if not fib_levels:
                continue
            for level, value in fib_levels.items():
                df.at[index, level] = value

    return df


def save_total_balance(exchange_id, total_balance, sub_account_id):
    date = str(datetime.now().date())  # Format: YYYY-MM-DD
    with trader_database.session_manager() as session:
        existing_record = session.query(BalanceReport).filter(
            BalanceReport.date == date,
            BalanceReport.exchange_id == exchange_id,
            BalanceReport.sub_account_id == sub_account_id
        ).one_or_none()

        if existing_record:
            # Update the existing record
            existing_record.value = total_balance
        else:
            # Create a new BalanceReport record
            balance_report_obj = BalanceReport(
                date=date,  # Ensure date is saved in YYYY-MM-DD format
                exchange_id=exchange_id,
                value=total_balance,
                sub_account_id=sub_account_id
            )
            session.add(balance_report_obj)
        session.commit()


def preprocess_oi(oi_df):
    if oi_df is not None:
        oi_df = oi_df.copy()
        keep_cols = ['timestamp', 'open_interest']
        oi_df['open_interest'] = round_series(oi_df['openInterestValue'], 0)
        oi_df = oi_df[keep_cols]
        oi_df['open_interest_sma_20'] = calculate_sma(oi_df, 20, column='open_interest')
        oi_df['open_interest_sma_10'] = calculate_sma(oi_df, 10, column='open_interest')
        oi_df.set_index('timestamp', inplace=True)
        return oi_df
    else:
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
    agent_id: str
    sub_account_id: str
    manual_side: str
    manual_standby_mode: bool

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
            agent_id=none_to_default(orm_obj.agent_id, None),
            pyramid_entry_limit=none_to_default(orm_obj.pyramid_entry_limit, 4),
            sub_account_id=none_to_default(orm_obj.sub_account_id, None),
            manual_side=none_to_default(orm_obj.manual_side, None),
            manual_standby_mode=none_to_default(orm_obj.manual_standby_mode, False)
        )


def load_strategy_settings(exchange_id, agent_id: str = None) -> List[StrategySettingsModel]:
    with (trader_database.session_manager() as session):
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
            StrategySettings.agent_id,
            StrategySettings.sub_account_id,
            StrategySettings.manual_side,
            StrategySettings.manual_standby_mode
        ).filter(
            StrategySettings.exchange_id == exchange_id,
            StrategySettings.active == True,
        )

        if agent_id:
            settings = settings.filter(StrategySettings.agent_id == agent_id)

        settings = settings.order_by(
            StrategySettings.timestamp_created
        ).all()

    # Convert the SQLAlchemy result rows to StrategySettingsModel using from_orm
    result_objects = [
        StrategySettingsModel.from_orm(row)
        for row in settings
    ]

    return result_objects


def process_new_shit_trader_strategy(exchange_id, ticker, timeframe, side, standby_mode):
    with trader_database.session_manager() as session:
        strategy_setting = session.query(StrategySettings).filter(
            StrategySettings.exchange_id == exchange_id,
            StrategySettings.ticker == ticker,
            StrategySettings.timeframe == timeframe,
            StrategySettings.manual_side == side,
            StrategySettings.manual_standby_mode == standby_mode,
            StrategySettings.agent_id == 'shit_trader'
        ).first()  # Retrieve the first matching record

        if strategy_setting:
            strategy_setting.active = True  # Update the 'active' attribute to True
            session.commit()  # Commit the changes to the database
        else:
            new_strategy = StrategySettings(
                exchange_id=exchange_id,
                ticker=ticker,
                timeframe=timeframe,
                buffer_days=20,
                stop_loss_atr_multipl=2,
                pyramid_entry_atr_multipl=1,
                aggressive_pyramid_entry_multipl=1,
                aggressive_price_atr_ratio=0,
                pyramid_entry_limit=1,
                active=True,
                agent_id='shit_trader',
                sub_account_id=None,
                manual_side=side,
                manual_standby_mode=standby_mode
            )
            _logger.info(f"Adding new shitcoin strategy: {new_strategy}")
            session.add(new_strategy)
            session.commit()
            _logger.info('strategy added')
