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

from src.config import TRADING_DATA_DIR, ATR_PERIOD, TURTLE_ENTRY_DAYS, TURTLE_EXIT_DAYS
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


def calculate_regression_channels_with_slope(df, periods=[20], upper_dev=2, lower_dev=2):
    """
    Calculate linear regression channels for multiple periods, including slope.

    Parameters:
    - df: DataFrame with 'timestamp' and 'C' columns.
    - periods: List of lookback periods to calculate regression channels.
    - upper_dev: Upper deviation factor.
    - lower_dev: Lower deviation factor.

    Returns:
    - Dictionary with period, direction, slope, upper, and lower bounds.
    """
    results = {}

    for n in periods:
        # Ensure we have enough data
        if len(df) < n:
            continue  # Skip periods longer than the dataset

        # Extract the last n rows
        window_df = df.iloc[-n:]
        time = np.arange(len(window_df))  # Use indices as time variable
        close_prices = window_df["C"]

        # Calculate Linear Regression Line
        m, b = np.polyfit(time, close_prices, 1)  # Slope and intercept
        regression_line = m * time + b

        # Calculate Standard Deviation of Residuals
        residuals = close_prices - regression_line
        std_dev = np.std(residuals)

        # Upper and Lower Bounds
        upper_bound = regression_line + upper_dev * std_dev
        lower_bound = regression_line - lower_dev * std_dev

        # Determine Direction
        direction = "up" if m > 0 else "down"

        # Save Results in Dictionary
        results[n] = {
            "period": n,
            "direction": direction,
            "slope": m,  # Slope of the trend
            "upper": upper_bound[-1],  # Last value of upper bound
            "lower": lower_bound[-1],  # Last value of lower bound
        }

    return results


def calculate_atr(df, period=ATR_PERIOD):
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
    atr = round_series(df['true_range'].rolling(window=period, min_periods=1).mean())

    # Clean up the DataFrame by removing the intermediate columns
    df.drop(['high_low', 'high_prev_close', 'low_prev_close', 'true_range'], axis=1, inplace=True)

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


def calculate_indicators_for_llm_trader(df):
    df['atr_20'] = calculate_atr(df, period=20)
    # df['atr_50'] = calculate_atr(df, period=50)
    df['sma_10'] = calculate_sma(df, period=10)
    df['sma_20'] = calculate_sma(df, period=20)
    df['sma_50'] = calculate_sma(df, period=50)
    df['sma_100'] = calculate_sma(df, period=100)
    # df['vol_sma_20'] = calculate_sma(df, period=20, column='V')
    # df['vol_sma_50'] = calculate_sma(df, period=50, column='V')
    df['rsi_14'] = calculate_rsi(df, period=14)
    df['rsi_sma_14'] = calculate_sma(df, period=14, column='rsi_14')
    df['macd_12_26'], df['macd_signal_9'] = calculate_macd(df)
    df['bollinger_band_middle_20'], df['bollinger_band_upper_20'], df[
        'bollinger_band_lower_20'] = calculate_bollinger_bands(df)
    # df['stochastic_k_14_s3'], df['stochastic_d_14_s3'] = calculate_stochastic_oscillator(df)
    df['adx_20'] = calculate_adx(df, n_periods=20)
    df['obv'] = round_series(calculate_obv(df), 0)
    df['obv_sma_20'] = round_series(calculate_sma(df, period=20, column='obv'), 0)
    return df


def calculate_indicators_for_llm_entry_validator(df):
    df = turtle_trading_signals_adjusted(df)
    df = calculate_indicators_for_llm_trader(df)
    df = df.drop(['short_exit', 'long_exit'], axis=1)
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
    """Identifies pivot highs and lows in a DataFrame based on ZigZag methodology."""
    atr: pd.Series = calculate_atr(df, period=depth)
    deviation_threshold = atr / df['C'] * (deviation / 100)

    high_deviation = df['H'] > (df['H'].shift(1) + deviation_threshold)
    low_deviation = df['L'] < (df['L'].shift(1) - deviation_threshold)

    pivot_highs = (df['H'].rolling(window=depth, center=True).max() == df['H']) & high_deviation
    pivot_lows = (df['L'].rolling(window=depth, center=True).min() == df['L']) & low_deviation

    df['pivot'] = np.where(pivot_highs, df['H'], np.where(pivot_lows, df['L'], np.nan))

    return df.dropna(subset=['pivot'])


def calculate_fib_levels_pivots(df, pivot_col='pivot', depth=20, deviation=2):
    """Calculate Fibonacci retracement levels from the pivot points."""
    df = df.copy()
    df = df[:-1]
    pivots = find_pivots(df, depth, deviation)
    last_pivot_high = pivots[pivots[pivot_col] == pivots['H']].last_valid_index()
    last_pivot_low = pivots[pivots[pivot_col] == pivots['L']].last_valid_index()

    if pd.notna(last_pivot_high) and pd.notna(last_pivot_low):
        pivot_high_price = pivots.at[last_pivot_high, 'H']
        pivot_low_price = pivots.at[last_pivot_low, 'L']

        # Determine trend
        if last_pivot_high < last_pivot_low:
            start_price = pivot_low_price
            end_price = pivot_high_price
        else:
            start_price = pivot_high_price
            end_price = pivot_low_price

        # Fibonacci levels calculation
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.272, 1.618]
        fib_levels = {f"fib_{level}": start_price + (end_price - start_price) * level for level in levels}
        fib_levels['swing_high'] = pivot_high_price
        fib_levels['swing_low'] = pivot_low_price
        fib_levels['depth'] = depth
        fib_levels['deviation'] = deviation
        return fib_levels

    _logger.warning(f"Count not calculate FIB dict")
    return None


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
    sub_account_id: str

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
            pyramid_entry_limit=none_to_default(orm_obj.pyramid_entry_limit, 4),
            sub_account_id=none_to_default(orm_obj.sub_account_id, None)
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
            StrategySettings.sub_account_id
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
