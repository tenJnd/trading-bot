from marshmallow import Schema, fields, post_load, EXCLUDE

from src.model.turtle_model import Order


class OrderSchema(Schema):
    id = fields.Str()
    client_order_id = fields.Str(data_key="clientOrderId", allow_none=True)
    timestamp = fields.Int(allow_none=True)
    datetime = fields.Str(allow_none=True)
    last_trade_timestamp = fields.Int(data_key="lastTradeTimestamp", allow_none=True)
    last_update_timestamp = fields.Int(data_key="lastUpdateTimestamp", allow_none=True)
    symbol = fields.Str(allow_none=True)
    type = fields.Str(allow_none=True)
    time_in_force = fields.Str(data_key="timeInForce", allow_none=True)
    post_only = fields.Bool(data_key="postOnly", allow_none=True)
    reduce_only = fields.Bool(data_key="reduceOnly", allow_none=True)
    side = fields.Str(allow_none=True)
    price = fields.Float(allow_none=True)
    trigger_price = fields.Float(allow_none=True, data_key="triggerPrice")
    amount = fields.Float(allow_none=True)
    cost = fields.Float(allow_none=True)
    average = fields.Float(allow_none=True)
    filled = fields.Float(allow_none=True)
    remaining = fields.Float(allow_none=True)
    status = fields.Str(allow_none=True)
    fee = fields.Dict(allow_none=True)
    trades = fields.List(fields.Dict(), allow_none=True)
    fees = fields.List(fields.Dict(), allow_none=True)
    stop_price = fields.Float(allow_none=True, data_key="stopPrice")
    take_profit_price = fields.Float(allow_none=True, data_key="takeProfitPrice")
    stop_loss_price = fields.Float(allow_none=True, data_key="stopLossPrice")
    info = fields.Dict(allow_none=True)

    exchange = fields.Str(allow_none=False)
    agg_trade_id = fields.Str(missing=None)

    atr = fields.Float(missing=None)
    position_status = fields.Str(missing='opened')
    action = fields.Str(missing=None)
    closed_positions = fields.List(fields.Str(), missing=None, allow_none=True)

    free_balance = fields.Float(missing=None)
    total_balance = fields.Float(missing=None)
    pl = fields.Float(missing=None)
    pl_percent = fields.Float(missing=None)

    contract_size = fields.Float(missing=1.0)
    strategy_id = fields.Integer(missing=None)

    class Meta:
        unknown = EXCLUDE

    @post_load
    def make_order(self, data, **kwargs):
        return Order(**data)
