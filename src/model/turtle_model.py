from sqlalchemy import Column, Float, String, Boolean, BigInteger, JSON, Numeric, ARRAY
from sqlalchemy.ext.declarative import declarative_base

from src.model import trader_database

Base = declarative_base()

SCHEMA = 'turtle_strategy'


class TurtleBase(Base):
    """Abstract DB model for all product tables"""
    __table_args__ = {'schema': SCHEMA}
    __abstract__ = True


class Order(TurtleBase):
    __tablename__ = 'orders'

    id = Column(String, primary_key=True)
    client_order_id = Column(String, index=True, nullable=True)
    timestamp = Column(BigInteger, nullable=True)
    datetime = Column(String, nullable=True)
    last_trade_timestamp = Column(BigInteger, nullable=True)
    last_update_timestamp = Column(BigInteger, nullable=True)
    symbol = Column(String, nullable=True)
    type = Column(String, nullable=True)
    time_in_force = Column(String, nullable=True)
    post_only = Column(Boolean, nullable=True)
    reduce_only = Column(Boolean, nullable=True)
    side = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    trigger_price = Column(Float, nullable=True)
    amount = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)
    average = Column(Float, nullable=True)
    filled = Column(Float, nullable=True)
    remaining = Column(Float, nullable=True)
    status = Column(String, nullable=True)
    fee = Column(JSON, nullable=True)
    trades = Column(JSON, nullable=True)
    fees = Column(JSON, nullable=True)
    stop_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    info = Column(JSON, nullable=True)

    exchange = Column(String, nullable=False)
    agg_trade_id = Column(String, nullable=False)

    atr = Column(Numeric, nullable=False)
    position_status = Column(String, nullable=False, default='opened')
    action = Column(String, nullable=False)
    closed_positions = Column(ARRAY(String), nullable=True)
    free_balance = Column(Float, nullable=True)
    total_balance = Column(Float, nullable=True)
    pl = Column(Float, nullable=True)
    pl_percent = Column(Float, nullable=True)


if __name__ == '__main__':
    trader_database.init_schema(Base.metadata)
