import json
import os
from dataclasses import dataclass
from typing import List

from src.config import TRADING_DATA_DIR
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


def load_strategy_settings(exchange_id) -> List[StrategySettingsModel]:
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
            StrategySettings.pyramid_entry_limit
        ).filter(
            StrategySettings.exchange_id == exchange_id,
            StrategySettings.active == True
        ).order_by(
            StrategySettings.timestamp_created
        ).all()

    # Convert the SQLAlchemy result rows to StrategySettingsModel using from_orm
    result_objects = [
        StrategySettingsModel.from_orm(row)
        for row in settings
    ]

    return result_objects
