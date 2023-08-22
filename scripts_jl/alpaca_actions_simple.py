
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import MarketOrderRequest,LimitOrderRequest
from datetime import datetime, timedelta
import time
import sys
from pytz import timezone
import alpaca_trade_api as tradeapi
import os
import pandas as pd
import numpy as np

PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/data"
ORDER_PATH = f"{PROJECT_PATH}/ACCOUNT"
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_jl"
ORDER_LOG_FILE = f"{SCRIPTS_PATH}/order_log.csv"
sys.path.append(PROJECT_PATH)

from scripts_jl.generate_ticker_list import choose_and_save_my_list, get_ticker_list
from scripts_jl.get_histories import download_histories, get_one_ticker_df
from scripts_jl.rs_rsi import compute_RSI
from scripts_jl.auto_arima import (arima_forecast)

########################################################################################
# this module is for simple trading. functions are used for your UI page
########################################################################################

END_POINT = "https://paper-api.alpaca.markets"


def get_trading_client(API_KEY, SECRET_KEY):
    """_summary_
        return a trade client
        input: api_key and secret, which are unique to your alpaca acounnt
        output: paper tradingclident

        Args:
            API_KEY (_type_): _description_
            SECRET_KEY (_type_): _description_

        Returns:
            _type_: _description_
    """
    return TradingClient(API_KEY, SECRET_KEY, paper=True)


def get_data_api(API_KEY  ,SECRET_KEY, END_POINT):
    """_summary_
        return resting api for getting stock data
        input: api_key and secret, which are unique to your alpaca acounnt
        output: api

        Args:
            API_KEY (_type_): _description_
            SECRET_KEY (_type_): _description_
            END_POINT (_type_): _description_

        Returns:
            _type_: _description_
    """
    return tradeapi.REST(API_KEY,SECRET_KEY, END_POINT)

def get_positions(trading_client):
    """_summary_
        get your account's positions, i.e., stocks you own
        inputer: your trading client
        output: position dataframe

        Args:
            trading_client (_type_): _description_

        Returns:
            _type_: _description_
    """
    positions = trading_client.get_all_positions()
    pos = []
    for position in positions:
        position = dict(position)
        pos.append((position["symbol"],position["exchange"].value, position["avg_entry_price"], position["current_price"],
            position["qty"] , position["side"].value ))

    return pd.DataFrame(pos, columns = ["Ticker", "Market", "Avg_Entry_Price", "Current_Price", "Qty", "Side"])


def get_account(trading_client):
    """_summary_
        get portfolio information
        input: trading client
        output: portfolio

        Args:
            trading_client (_type_): _description_

        Returns:
        _type_: _description_
    """
    account = trading_client.get_account()
    account = dict(account)
    portfolio_value = '{:.2f}'.format(float(account["portfolio_value"]))
    return account["account_number"], account["buying_power"], portfolio_value

def get_history(api, ticker, interval,  data_points=250):
    """
        use alpaca api to get near real time stock trade data (15min delay)
        intervals: 59Min, 23Hour, 1Day, 1Week, 12Month .
        from_time, to_time. to_time default to None. in format of "2023-02-12 12:00"
        minimum requires 250 data points.take some extra
        for precise, need to write a function

        output: ohlc datafraem
    """
    ticker = ticker.upper()
    interval = '1Day' if interval=='1 day' else '1Hour' if interval=='1 hour' else '5Min' if interval=='5 min' else '15Min'

    if interval== '1Day':
        from_time = datetime.utcnow() + timedelta(days=-450)
        data_points = 300
    elif interval == '1Hour':
        from_time = datetime.utcnow() + timedelta(days=-20)
        data_points = 20*24
    elif interval == '5Min':
        from_time = datetime.utcnow() + timedelta(days=-6)
        data_points = 16 * 12 * 6
    else: #15in
        from_time = datetime.utcnow() + timedelta(days=-15)
        data_points = 16 * 4 * 15

    from_time = from_time.strftime('%Y-%m-%d')
    ticker = ticker.upper()
    stock_data = api.get_bars(ticker, timeframe=interval, start=from_time, limit=data_points).df
    stock_data.reset_index(inplace=True)
    #stock_data = api.get_bars(ticker, timeframe="1Day", start="2023-02-01", limit=250).df
    stock_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Trades", "VWap"]

    return stock_data

def get_history_brief(api,ticker, interval,  data_points=250):
    """
        intervals: 59Min, 23Hour, 1Day, 1Week, 12Month .
        from_time, to_time. to_time default to None. in format of "2023-02-12T12:00"

        output: get only the most recent data points
    """
    ticker = ticker.upper()
    interval = '1Day' if interval=='1 day' else '1Hour' if interval=='1 hour' else '5Min' if interval=='5 min' else '15Min'
    ticker = ticker.upper()
    stock_data = api.get_bars(ticker, timeframe=interval,  limit=data_points).df
    stock_data.reset_index(inplace=True)
    #stock_data = api.get_bars(ticker, timeframe="1Day", start="2023-02-01", limit=250).df
    stock_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Trades", "VWap"]
    return stock_data

def get_orders(trading_client, status = 'closed'):
    """_summary_

    list accounts orders
    output: account orders' dataframe
    Args:
        API_KEY (_type_): _description_
        SECRET_KEY (_type_): _description_
        status (str, optional): _description_. Defaults to 'closed'.

    Returns:
        _type_: _description_
    """
    orders = trading_client.list_orders(
        status=status,
        limit=100,
        nested=True  # show nested multi-leg orders
    )

    results = []
    for oo in orders:
        results.append((oo.client_order_id, oo.symbol, oo.created_at,oo.filled_at, oo.filled_avg_price, \
            oo.filled_qty, oo.id, oo.order_type, oo.status))

    return pd.DataFrame(results, columns=["client_order_id", 'ticker', "created_at", "filled_at", "filled_avg_price",\
                    "filled_qty",   "id", "order_type", "status"])


def submit_alpaca_order_simple(
                        trading_client,
                        api,
                        model = "RSI",
                        action = 'buy',
                        interval = '1 day',
                        ticker = 'AAPL',
                        order_type = 'Limit',
                        order_valid = 'GTC',
                        amount = None,
                        rsi = 20,
                        price = 345,
                        shares = 40
                    ):
    """_summary_
    goal:
        submit order directly to alpaca.
        input: as args below
        output: if order is valid, will return successful msg. otherwise error msg.

    Args:
        model (_type_, optional): _description_. Defaults to sell_model.
        action (str, optional): _description_. Defaults to 'buy'.
        interval (_type_, optional): _description_. Defaults to sell_model_granular.
        ticker (_type_, optional): _description_. Defaults to sell_ticker.
        order_type (_type_, optional): _description_. Defaults to sell_type.
        order_valid (_type_, optional): _description_. Defaults to sell_enforce.
        amount (_type_, optional): _description_. Defaults to None.
        rsi (_type_, optional): _description_. Defaults to sell_rsi.
        price (_type_, optional): _description_. Defaults to sell_price.
        shares (_type_, optional): _description_. Defaults to sell_shares.
    """

    # trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    # api = tradeapi.REST(API_KEY  ,SECRET_KEY, END_POINT)

    valid, msg = validate_order_simple_(trading_client, ticker, action, price,shares)
    if valid == 1:
        return 1, msg

    side = OrderSide.BUY if action.lower()=='buy' else OrderSide.SELL
    enforce = TimeInForce.DAY if order_valid.lower()=='day' else TimeInForce.GTC

    if order_type == 'Market':
        market_order_data = MarketOrderRequest( symbol=ticker, qty=shares, side=side, time_in_force=enforce)
    else: # Limit
        market_order_data = LimitOrderRequest( symbol=ticker, qty=shares, side=side, type=OrderType.LIMIT,
                                            limit_price=price, time_in_force=enforce)

    order_ = trading_client.submit_order( order_data=market_order_data )
    customer_id = order_.id
    alpaca_status = 'submited'

    return 0, f"Successfully submitted the order. Order Id: {customer_id}. Alpaca Order Status: {alpaca_status}"



def  validate_order_simple_(trading_client, ticker, action, price,shares):
    """_summary_
    validate:
        1. whether enough fund
        2. calculate round number of shares
        3. generate fail/success message
    Args:
        model (_type_): _description_
        action (_type_): _description_
        interval (_type_): _description_
        ticker (_type_): _description_
        order_type (_type_): _description_
        order_valid (_type_): _description_
        amount (_type_): _description_
        rsi (_type_): _description_
        price (_type_): _description_
        shares (_type_): _description_
    """
    _, buying_power, _ = get_account(trading_client)
    if action.lower() == 'sell':
        df_position = get_positions(trading_client)
        shares_ = 0
        df_ = df_position.query(f"Ticker=='{ticker}'")
        if not df_.empty:
            shares_ = df_["Qty"][0]
        if shares_ < shares:
            return 1,  f"You don't have enough shares of {ticker}. You have {shares_}"
        else:
            return 0, ""
    else:
        if price*shares > float(buying_power):
            return 1,  f"You don't have enough fund for the transaction . You have {buying_power}. But order requires {price*shares}"
        elif price*shares == 0:
            return 1,  f"Both shares and buy price can not be zero"
        else:
            return 0, ""

def get_rsi(api, ticker, interval, price_type='Close', span=14, min_periods = 3, recent_n = 5 ):
    """_summary_
    get recent rsi readings for ticker
    input: as below.
    output: recent_n rsi readings
    Args:
        api (_type_): _description_
        ticker (_type_): _description_
        interval (_type_): _description_
        price_type (str, optional): _description_. Defaults to 'Close'.
        span (int, optional): _description_. Defaults to 14.
        min_periods (int, optional): _description_. Defaults to 3.
        recent_n (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    df = get_history(api, ticker, interval,  data_points=250)

    price_data = df[price_type]
    diff = price_data.diff(1).dropna()
    up_chg = 0 * diff #arrays
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[ diff>0 ]
    down_chg[diff < 0] = diff[ diff < 0 ]

    up_series_avg   = up_chg.ewm(span=span , min_periods=min_periods).mean()
    down_series_avg = down_chg.ewm(span=span , min_periods=min_periods).mean()

    rs = abs(up_series_avg/down_series_avg)
    rsi = 100 - 100/(1+rs)

    df['Up'] =up_series_avg
    df['Down'] =down_series_avg
    df[f'RS_{span}'] =rs
    df[f'RSI_{span}'] =rsi

    return list(df[f'RSI_{span}'])[-recent_n:]


def get_arima(api, ticker, interval, price_type='Close', plot_percentage_change=False,predict_n=5):
    """_summary_
    call arima forecast method from auto_arima module to get predictions.
    input: as follows
    output: future predict_n forecasts
    Args:
        api (_type_): _description_
        ticker (_type_): _description_
        interval (_type_): _description_
        price_type (str, optional): _description_. Defaults to 'Close'.
        plot_percentage_change (bool, optional): _description_. Defaults to False.
        predict_n (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    df = get_history(api, ticker, interval,  data_points=250)
    if df.shape[0] < 30:
        return [-1]*predict_n # not enough data for forecasting. silently return
    forecast = arima_forecast(df, price_type=price_type, plot_percentage_change=plot_percentage_change, predict_n=predict_n)
    return forecast
