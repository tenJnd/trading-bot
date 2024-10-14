""" simple script for stop-loss calculation """


def calculate_stop_loss_based_on_risk(position, capital, risk_percent, asset_price, leverage=1):
    risk_amount = capital * risk_percent
    leveraged_capital = capital * leverage
    price_move = (risk_amount / leveraged_capital) * asset_price

    if position == 'long':
        stop_loss_price = asset_price - price_move
    else:
        stop_loss_price = asset_price + price_move

    asset_amount = leveraged_capital / asset_price
    loss = leveraged_capital - (asset_amount * stop_loss_price)
    assert round(abs(loss), 1) == round(risk_amount, 1)

    print(f"============\n"
          f"Find stop-loss price if investing {capital} with leverage {leverage} "
          f"and risking {risk_percent} of capital\n"
          f"position {str.upper(position)} set up:\n"
          f"capital: {capital}\n"
          f"leverage: {leverage}\n"
          f"percent risk of cap: {risk_percent * 100}%\n"
          f"risk amount: {risk_amount}\n"
          f"leverage entry: amount {asset_amount}\n"
          f"stop-loss price: {stop_loss_price}")

    return stop_loss_price


def calculate_position_size_based_on_move_against(position, capital, asset_price, move, risk_percent=0.01, leverage=1):
    leveraged_risk = leverage * risk_percent

    risk_amount = leveraged_risk * capital
    asset_amount = risk_amount / move
    inv = asset_amount * asset_price

    if position == 'long':
        stop_loss_price = asset_price - move
    else:
        stop_loss_price = asset_price + move

    check = inv - (stop_loss_price * asset_amount)
    assert round(abs(check), 1) == round(risk_amount, 1)

    print(f"============\n"
          f"Find position size based on price move ({move} against the position), "
          f"risk percent and leverage. Risk percent of capital "
          f"is multiplied be leverage!!\n"
          f"position {str.upper(position)} set up:\n"
          f"capital: {capital}\n"
          f"leverage: {leverage} | "
          f"risk unit percent: {risk_percent}\n"
          f"percent risk of cap: {leveraged_risk * 100}%\n"
          f"risk amount: {risk_amount}\n"
          f"leverage entry: amount {asset_amount}, investment {inv}\n"
          f"stop-loss price: {stop_loss_price}")

    return asset_amount


def calculate_risk_percent_based_on_stop_loss(position, capital, asset_price, stop_loss_price, leverage=1):
    leveraged_capital = capital * leverage
    asset_amount = leveraged_capital / asset_price

    if position == 'long':
        price_move = asset_price - stop_loss_price
    else:
        price_move = stop_loss_price - asset_price

    loss = asset_amount * price_move
    risk_amount = leveraged_capital - (leveraged_capital - loss)
    risk_percent = risk_amount / capital

    print(f"============\n"
          f"Calculate the risk percentage based on stop-loss price\n"
          f"position {str.upper(position)} set up:\n"
          f"capital: {capital}\n"
          f"leverage: {leverage}\n"
          f"asset price: {asset_price}\n"
          f"stop-loss price: {stop_loss_price}\n"
          f"risk amount: {risk_amount}\n"
          f"percent risk of capital: {round(risk_percent * 100, 1)}%")

    return risk_percent


if __name__ == '__main__':
    position = 'long'
    asset_price = 400.5
    capital = 2770
    leverage = 1
    risk_percent = 0.01

    # function specific
    move = 11.93*2
    stop_loss = 362

    calculate_stop_loss_based_on_risk(position=position,
                                      capital=capital,
                                      risk_percent=risk_percent,
                                      asset_price=asset_price,
                                      leverage=leverage)

    calculate_position_size_based_on_move_against(position=position,
                                                  capital=capital,
                                                  asset_price=asset_price,
                                                  move=move,
                                                  risk_percent=risk_percent,
                                                  leverage=leverage  # percent of capital risk
                                                  )

    calculate_risk_percent_based_on_stop_loss(position=position,
                                              capital=capital,
                                              asset_price=asset_price,
                                              stop_loss_price=stop_loss,
                                              leverage=leverage)

