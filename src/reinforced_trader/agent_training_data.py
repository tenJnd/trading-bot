from datetime import datetime

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.utils import (calculate_indicators_for_llm_trader)


def transform_candle_stats(df: pd.DataFrame) -> pd.DataFrame:
    df['day_num'] = df['datetime'].dt.day
    df['date'] = df['datetime'].dt.date
    df['date'] = df['date'].apply(
        lambda x: datetime.strftime(x, "%Y%m%d")
    ).astype(int)

    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['week_num'] = df['datetime'].dt.isocalendar().week
    df['year_num'] = df['datetime'].dt.isocalendar().year
    df['year_day'] = (df['year_num'].astype(str) +
                      df['day_num'].astype(str))
    df['year_week'] = (df['year_num'].astype(str) +
                       df['week_num'].astype(str))

    df = calculate_indicators_for_llm_trader(df)
    return df


def prepare_training_data(exchange):
    ohlc = exchange.fetch_ohlc(days=365, timeframe='1h')
    ohcl_df = transform_candle_stats(ohlc)

    scaler = MinMaxScaler()
    ohcl_df.dropna(inplace=True)

    # Storing original data
    original = ohcl_df[['timeframe', 'C', 'date', 'year_week', 'atr_20']]
    original_groups = original.groupby('date')
    groups_len = set(len(group) for name, group in original_groups)
    groups_max_len = max(groups_len)
    original_groups = [(name, group) for name, group in original_groups if not len(group) < groups_max_len]

    # Dropping columns not needed for scaling
    transformed_data_for_scaling = ohcl_df.drop(columns=['year_week', 'year_day', 'datetime'], axis=1)

    # Scaling the data
    scaled_data_array = scaler.fit_transform(transformed_data_for_scaling)

    # Saving the scaler
    joblib.dump(scaler, 'mm_scaler.gz')

    # Converting scaled data back to DataFrame
    scaled_data = pd.DataFrame(scaled_data_array, columns=transformed_data_for_scaling.columns,
                               index=transformed_data_for_scaling.index)
    scaled_groups = scaled_data.groupby('date')
    scaled_groups = [(name, group) for name, group in scaled_groups if not len(group) < groups_max_len]
    zipped = zip(scaled_groups, original_groups)

    return zipped, len(list(scaled_data.columns)), len(original_groups)
