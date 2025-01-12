import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.utils import calculate_indicators_for_nn


def transform_candle_stats(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Basic date-time components
    df['day_num'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['week_of_year'] = df['datetime'].dt.isocalendar().week

    # Seasonal transformations
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['datetime'].dt.dayofyear / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)

    # Dynamic grouping based on input
    if group_by == 'day':
        df['group'] = df['datetime'].dt.strftime('%Y%m%d').astype(int)
    elif group_by == 'yearweek':
        df['group'] = df['datetime'].dt.strftime('%G%V')
    elif group_by == 'month':
        df['group'] = df['datetime'].dt.strftime('%Y%m')
    else:
        raise ValueError(f"Unsupported group_by value: {group_by}")

    df = calculate_indicators_for_nn(df)
    return df


def prepare_training_data(exchange, days: int, timeframe: str, group_by: str):
    ohlc = exchange.fetch_ohlc(days=days, timeframe=timeframe)
    ohcl_df = transform_candle_stats(ohlc, group_by)

    scaler = MinMaxScaler()
    ohcl_df.dropna(inplace=True)

    original = ohcl_df[['timeframe', 'C', 'H', 'L', 'group', 'atr_20']]
    original_groups = original.groupby('group')

    groups_len = set(len(group) for _, group in original_groups)
    groups_max_len = max(groups_len)
    original_groups = [(name, group) for name, group in original_groups if len(group) == groups_max_len]

    columns_to_scale = ohcl_df.drop(columns=['group', 'datetime'], axis=1).columns
    scaled_data_array = scaler.fit_transform(ohcl_df[columns_to_scale])
    joblib.dump(scaler, f'mm_scaler_{timeframe}.gz')

    scaled_data = pd.DataFrame(scaled_data_array, columns=columns_to_scale, index=ohcl_df.index)
    scaled_groups = scaled_data.groupby(ohcl_df['group'])
    scaled_groups = [(name, group) for name, group in scaled_groups if len(group) == groups_max_len]
    zipped = zip(scaled_groups, original_groups)

    return zipped, len(columns_to_scale), len(original_groups)
