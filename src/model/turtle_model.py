from datetime import datetime, timezone

from sqlalchemy import Column, Float, String, Boolean, BigInteger, JSON, Numeric, ARRAY, Integer, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy_utc import UtcDateTime

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

    contract_size = Column(Float, nullable=True)
    strategy_id = Column(Integer, ForeignKey('turtle_strategy.strategy_settings.id'), nullable=True)
    atr_period_ratio = Column(Float, nullable=False, default=1.0)
    candle_timeframe = Column(BigInteger, nullable=True)

    # Relationship to StrategySettings
    strategy = relationship("StrategySettings", back_populates="orders")


class StrategySettings(TurtleBase):
    __tablename__ = 'strategy_settings'

    id = Column(Integer, primary_key=True)
    exchange_id = Column(String)
    ticker = Column(String)
    timeframe = Column(String)
    buffer_days = Column(Integer)
    stop_loss_atr_multipl = Column(Float)  # multiplying ATR for stop-loss price
    pyramid_entry_atr_multipl = Column(Float)  # Multiplying ATR for pyramid entry
    aggressive_pyramid_entry_multipl = Column(Float)  # multiply ATR for pyramid entry when price/ratio con is met
    aggressive_price_atr_ratio = Column(Float)  # price/atr ration when we use aggressive entry (with lowe ATR)
    pyramid_entry_limit = Column(Integer)  # How many pyramid entries we can do for one asset (4 = 1 init, 3 pyramid)
    timestamp_created = Column(UtcDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    active = Column(Boolean, default=False)
    agent_id = Column(String)
    sub_account_id = Column(String)

    # Relationship to Order
    orders = relationship("Order", back_populates="strategy")
    agent_actions = relationship("AgentActions", back_populates="strategy")


class DepositsWithdrawals(TurtleBase):
    __tablename__ = 'deposits_withdrawals'

    id = Column(Integer, primary_key=True)
    exchange_id = Column(String)
    timestamp_created = Column(UtcDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    value = Column(Float)
    sub_account_id = Column(String, default=None)
    person_id = Column(String)


class BalanceReport(TurtleBase):
    __tablename__ = 'balance_report'

    id = Column(Integer, primary_key=True)
    exchange_id = Column(String)
    timestamp_created = Column(UtcDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    date = Column(String)
    value = Column(Float)
    sub_account_id = Column(String, default=None)


class TurtleBackTest(TurtleBase):
    __tablename__ = 'turtle_back_test'

    id = Column(Integer, primary_key=True)
    exchange_id = Column(String)
    ticker = Column(String)
    init_capital = Column(Float)
    pl = Column(Float)
    pl_percent = Column(Float)
    final_capital = Column(Float)
    trades = Column(ARRAY(Float))
    timestamp_created = Column(UtcDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    timeframe = Column(String)
    period_days = Column(Integer)


class AgentActions(TurtleBase):
    __tablename__ = 'agent_actions'

    id = Column(Integer, primary_key=True)
    action = Column(String)
    rationale = Column(Text)
    agent_output = Column(JSON)
    strategy_id = Column(Integer, ForeignKey('turtle_strategy.strategy_settings.id'), nullable=True)
    timestamp_created = Column(UtcDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    order = Column(JSON, nullable=True)
    candle_timestamp = Column(BigInteger)
    agent_name = Column(String)

    strategy = relationship("StrategySettings", back_populates="agent_actions")


class EpisodesTraining(TurtleBase):
    __tablename__ = 'episodes_training'

    episode_group = Column(String, primary_key=True)
    balance = Column(Float)
    total_reward = Column(Float)
    total_profit = Column(Float)
    win_trades = Column(Integer)
    lost_trades = Column(Integer)


if __name__ == '__main__':
    trader_database.init_schema(Base.metadata)
