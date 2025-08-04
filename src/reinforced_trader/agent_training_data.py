from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.utils import calculate_indicators_for_nn_4h, calculate_indicators_for_nn_30m


def transform_candle_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms OHLC data by adding cyclic date-time features."""
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Seasonal transformations (sin/cos encoding for periodicity)
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['datetime'].dt.dayofyear / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)

    return df


def load_and_process_ohlc_data(exchange, ticker: str, days: int, timeframe: str):
    """Loads OHLC data, applies transformations and indicators."""
    dataset_file_name = Path(f"ohlc_{ticker}_{days}_{timeframe}.csv")

    if not dataset_file_name.exists():
        ohlc = exchange.fetch_ohlc(days=days, timeframe=timeframe)
        ohlc.to_csv(dataset_file_name, index=False)
    else:
        ohlc = pd.read_csv(dataset_file_name)

    # Apply transformations and indicators
    ohlc = transform_candle_stats(ohlc)
    ohlc = calculate_indicators_for_nn_4h(ohlc)
    ohlc.dropna(inplace=True)

    return ohlc


def prepare_training_data(exchange, ticker, days: int, timeframe: str):
    """Prepares training data without grouping, keeping the entire dataset as one sequence."""
    scaler_filename = Path(f'scaler_{ticker}_{days}_{timeframe}.gz')

    # Load processed OHLC data
    ohlc_df = load_and_process_ohlc_data(exchange, ticker, days, timeframe)

    # Save essential columns (unscaled)
    original_data = ohlc_df[['datetime', 'C', 'H', 'L', 'atr_20', 'atr_50']].copy()

    # Select features to scale (exclude non-numerical and unnecessary columns)
    columns_to_scale = ohlc_df.drop(columns=['timeframe', 'datetime'], axis=1).columns

    # Load or create scaler
    if not scaler_filename.exists():
        scaler = MinMaxScaler()
        scaled_data_array = scaler.fit_transform(ohlc_df[columns_to_scale])
        joblib.dump(scaler, scaler_filename)
    else:
        scaler = joblib.load(scaler_filename)
        scaled_data_array = scaler.fit_transform(ohlc_df[columns_to_scale])

    # Create scaled DataFrame
    scaled_data = pd.DataFrame(scaled_data_array, columns=columns_to_scale, index=ohlc_df.index)

    return scaled_data, original_data
