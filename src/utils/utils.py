import json
import os
from dataclasses import dataclass

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


def save_json_to_file(order_data: OrderSchema, file_name: str):
    # Convert the order data to a dictionary
    order_dict = order_data.dump(order_data)

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


@dataclass
class StrategySettingsModel:
    exchange_id: str
    ticker: str
    timeframe: str
    buffer_days: int


def load_strategy_settings(exchange_id) -> StrategySettingsModel:
    with trader_database.session_manager() as session:
        # Querying the database
        settings = session.query(
            StrategySettings.exchange_id,
            StrategySettings.ticker,
            StrategySettings.timeframe,
            StrategySettings.buffer_days
        ).filter(
            StrategySettings.exchange_id == exchange_id,
        ).order_by(
            StrategySettings.timestamp_created
        )

        # Create instances of StrategySettingsModel for each result
        result_objects = [
            StrategySettingsModel(
                exchange_id=row.exchange_id,
                ticker=row.ticker,
                timeframe=row.timeframe,
                buffer_days=row.buffer_days
            )
            for row in settings.all()
        ]

    return result_objects
