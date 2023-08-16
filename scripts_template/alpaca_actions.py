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
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_template"
ORDER_LOG_FILE = f"{SCRIPTS_PATH}/order_log.csv"
sys.path.append(PROJECT_PATH)

from scripts_template.generate_ticker_list import choose_and_save_my_list, get_ticker_list
from scripts_template.get_histories import download_histories, get_one_ticker_df
from scripts_template.rs_rsi import compute_RSI
from utility import DataProcessing
from scripts_template.auto_arima import (arima_forecast)


END_POINT = "https://paper-api.alpaca.markets"

# API_KEY = "PKYHZ0L6101MSFJDO3QN"
# SECRET_KEY = "EdqZW5yHPSRdgOOnREghIQEsvCz6O7i5LpMje0jJ"

def get_trading_client(API_KEY, SECRET_KEY):
    return TradingClient(API_KEY, SECRET_KEY, paper=True)
def get_data_api(API_KEY  ,SECRET_KEY, END_POINT):
    return tradeapi.REST(API_KEY,SECRET_KEY, END_POINT)

def get_positions(trading_client):
    positions = trading_client.get_all_positions()
    pos = []
    for position in positions:
        position = dict(position)
        pos.append((position["symbol"],position["exchange"].value, position["avg_entry_price"], position["current_price"],
            position["qty"] , position["side"].value ))

    return pd.DataFrame(pos, columns = ["Ticker", "Market", "Avg_Entry_Price", "Current_Price", "Qty", "Side"])

    #pp = get_positions()


def get_account(trading_client): #API_KEY, SECRET_KEY):
    #trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    account = trading_client.get_account()
    account = dict(account)
    return account["account_number"], account["buying_power"], account["portfolio_value"]

def get_pending_orders():
    columns = ['ticker','order number', 'action', 'model', 'order date(utc)', 'ordertype', 'shares', 'amount', 'rsi_value', 'price']
    return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

def get_history(api, ticker, interval,  data_points=250):
    """
        intervals: 59Min, 23Hour, 1Day, 1Week, 12Month .
        from_time, to_time. to_time default to None. in format of "2023-02-12 12:00"
        minimum requires 250 data points.take some extra
        for precise, need to write a function
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
    """
    ticker = ticker.upper()
    interval = '1Day' if interval=='1 day' else '1Hour' if interval=='1 hour' else '5Min' if interval=='5 min' else '15Min'
    ticker = ticker.upper()
    stock_data = api.get_bars(ticker, timeframe=interval,  limit=data_points).df
    stock_data.reset_index(inplace=True)
    #stock_data = api.get_bars(ticker, timeframe="1Day", start="2023-02-01", limit=250).df
    stock_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Trades", "VWap"]
    return stock_data

def get_orders(API_KEY, SECRET_KEY, status = 'closed'):
    # status='open'
    api = tradeapi.REST(API_KEY  ,SECRET_KEY, END_POINT)
    orders = api.list_orders(
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

def cancel_order(API_KEY, SECRET_KEY, order_id):
    """
        order_id: client order id
    """
    api = tradeapi.REST(API_KEY  ,SECRET_KEY, END_POINT)
    api.cancel_order(order_id = order_id )
    return 0


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


def submit_alpaca_order(
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
    goal: add to the order queue.
        start: validation
        if market order:
            execute right away, get the customer order id
            add to order_log
            market order as executed
            doesn't matter it is rsi or arima
        if limit order:
            if rsi:
                run rsi value
                if meet:
                    execute order as market order and get customer order id
                    add to order_log
                    mark order as executed.
                if not:
                    add to order_log
                    get an order_id which is NOT alpaca order id. rather internal id
                    mark order as logged
            else if arima:
                run near realtime api to get the price
                    if meet:
                        execute order as market order and get customer order id
                        add to order_log
                        mark order as executed.
                    if not:
                        submit the order to alpaca. get customer order id
                        add to order_log
                        mark order as submitted

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
    #valid, msg, shares_ = validate_order_(trading_client, model, action, interval,ticker, rsi,price,shares )
    valid, msg, shares_ = validate_order_(trading_client, model, action,  ticker,  price,shares )
    if not valid:
        return 1, msg

    sp_id = str(int(time.mktime(datetime.datetime.utcnow().timetuple())))

    to_submit_n_log = True # False: log only
    if order_type == 'Market':
        to_submit_n_log = True
    elif order_type == "Limit":
        if model == 'RSI':
            rsi_value = get_rsi(ticker, interval, price_type='Close')
            if (action.lower()=='buy' and rsi_value<=rsi) or (action.lower()=='sell' and rsi_value>=rsi):
                to_submit_n_log = True
            else:
                to_submit_n_log = False
        else:
            to_submit_n_log = True
            price_now = get_arima(ticker, interval, price_type='Close')
            if (action.lower()=='buy' and price_now<=price) or (action.lower()=='sell' and price_now>=price):
                order_type = 'Market'   # make it to a market order by execute right away

    submit_order_(to_submit_n_log, trading_client, model, action, interval,ticker,order_type,order_valid,
                  amount,rsi,price,shares,sp_id, shares_ )

def submit_order_(trading_client, to_submit_n_log, model, action, interval,ticker,order_type,order_valid,
        amount,rsi,price,shares,sp_id, shares_ ):

    customer_id, sp_status, alpaca_status = None, 'logged', 'logged'
    if not to_submit_n_log:
        side = OrderSide.BUY if action.lower()=='buy' else OrderSide.SELL
        enforce = TimeInForce.DAY if order_valid.lower()=='day' else TimeInForce.GTC
        if amount > 0:
            shares = shares_

        if order_type == 'Market':
            market_order_data = MarketOrderRequest( symbol=ticker, qty=shares, side=side, time_in_force=enforce)
        else: # Limit
            market_order_data = LimitOrderRequest( symbol=ticker, qty=shares, side=side, type=OrderType.LIMIT,
                                                limit_price=price, time_in_force=enforce)

        order_ = trading_client.submit_order( order_data=market_order_data )
        customer_id = order_.id
        alpaca_status = 'submited'

    log_alpaca_order_(model, action, interval,ticker,order_type,order_valid,amount,rsi,price,shares,
                        sp_id=sp_id, alpaca_id=customer_id, sp_status=sp_status, alpaca_status=alpaca_status)


def  validate_order_(trading_client, model, action, ticker,price,shares):
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
            if shares > shares_:
                return 1, 'Not shares', shares_
            else:
                return 0, '', shares_
        else:
            return 1, 'No shares', 0
    else: # buy
        if model == 'RSI':
            return 0, buying_power, None
        else:
            return 0, '', int(buying_power/price)


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
    df = get_history(api, ticker, interval,  data_points=250)
    forecast = arima_forecast(df, price_type=price_type, plot_percentage_change=plot_percentage_change, predict_n=predict_n)
    return forecast


def log_alpaca_order_(model, action, interval,ticker,order_type,order_valid,amount,rsi,price,shares, sp_status, alpaca_status):

    new_row = {"model": [model], "action": [action], "interval":[interval], "ticker":[ticker], "order_type":[order_type],
               "order_valid":[order_valid], "amount":[amount], "rsi":[rsi], "price":[price], "shares":[shares],
               "sp_status":[sp_status], "alpaca_status":[alpaca_status]}
    if os.path.exists(ORDER_LOG_FILE):
        df = pd.read_csv(ORDER_LOG_FILE)
        df = df.append(new_row, ignore_index=True)
    else:
        df = pd.DataFrame(new_row)

    df.to_csv(ORDER_LOG_FILE, index=False)


def alpaca_order_execution(trading_client,):
    """
        go to the pre-order data set to check out whether trading conditions met
        once executed, remove it from the queque
        submit order as market order
        for order that alpaca_status=='logged'

    """
    PRE_ORDERS = f"{ORDER_PATH}/pre_orders.csv"

    if not os.path.exists(PRE_ORDERS):
        return
    df = pd.read_csv(PRE_ORDERS)
    if df.empty:
        return

    for row in df.itertuples():
        row_id = row.Index
        model = row.sell_model,
        action = row.action,
        interval = row.model_granular,
        ticker = row.ticker,
        order_type = row.type,
        order_valid = row.enforce,
        amount = row.amount,
        rsi = row.rsi,
        price = row.price,
        shares = row.shares
        sp_id = row.sp_id
        alpaca_status = row.alpaca_status

        if alpaca_status != 'logged':
            continue


        # first check model type.
        # when submit order, automatically loggoed.
        to_submit_n_log = True
        updated = False
        if model == "RSI":
            # 1st - calculate RSI
            # if meet requirements, then execute
            rsi_ = get_rsi()
            if rsi > rsi_ and action.lower()=="sell":
                submit_order_(trading_client, to_submit_n_log, model, action, interval,ticker,order_type,order_valid,
                    amount,rsi,price,shares,sp_id, shares )
                updated = True
            elif rsi < rsi_ and action.lower() == "buy":
                submit_order_(trading_client, to_submit_n_log, model, action, interval,ticker,order_type,order_valid,
                amount,rsi,price,shares,sp_id, shares )
                updated = True
        else:
            arima_ = get_arima()
            if price > arima_ and action.lower()=="buy":
                submit_order_(trading_client, to_submit_n_log, model, action, interval,ticker,order_type,order_valid,
                    amount,rsi,price,shares,sp_id, shares )
                updated = True
            elif price < arima_ and action.lower()=="sell":
                submit_order_(trading_client, to_submit_n_log, model, action, interval,ticker,order_type,order_valid,
                    amount,rsi,price,shares,sp_id, shares )
                updated = True
        if updated:
            df.loc[row_id, 'sp_status'] = 'submitted'
            df.loc[row_id, 'alpaca_status'] = 'submitted'
