
import os
import sys
import re
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/data"
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_template"
sys.path.append(PROJECT_PATH)

from scripts_template.auto_arima import train_autoarima_for_batch
from scripts_template.generate_ticker_list import get_ticker_list
from scripts_template.trade_rsi_strategy import create_position,  implement_rsi_strategy


def load_performance(model_type="RSI"):
    output_file = f"{PROJECT_PATH}/Ticker_performance_{model_type.lower()}.csv"
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    return pd.DataFrame()

# plot a stock's OHLC chart
def rsi_model():
    # let create a performance file under root folder
    output_file = f"{PROJECT_PATH}/Ticker_performance_rsi.csv"
    # files headers: ["ticker", "interval", "price_type", "starting", "ending", "performance"]
    # to simplify, every time we retrain, simply overwrite.
    # add trained time at the end
    interval = '1d'
    price_type = 'Close'
    starting_fund = 2000000
    tickers = get_ticker_list()

    # we use try except here !!!
    # reason: some ticker may not exist; some possible implete data; etc.
    # we don't handle exception. simple pass
    # output list of tuples
    performances = []
    for ticker in tickers:
        try:
            buy_price, sell_price, rsi_signal, df_ =\
                implement_rsi_strategy(ticker=ticker, interval=interval, price_type=price_type)
            df_position =  create_position(ticker=ticker, interval=interval,
                rsi_signal=rsi_signal, price_type=price_type,  starting_fund=starting_fund )

            trades = df_position[df_position["Position_Signal"]!=0].shape[0]
            starting = df_position.iloc[0][-1]
            ending = df_position.iloc[-1][-1]
            ticker, ending, starting, trades
            change = -100 if ending==0 else (ending-starting)/starting *100
            performances.append((ticker, interval, price_type, starting, ending, change))

        except:
            pass
    df_result = pd.DataFrame(performances,
        columns = ["ticker", "interval", "price_type", "starting", "ending", "performance"] )
    df_result["trained_on"] =datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M")

    # ranking
    df_result["rank"] = df_result["performance"].rank(ascending=False, method="first").astype(int)
    df_result.sort_values(by="rank", ascending=True, inplace=True)
    df_result.to_csv(output_file)


def arima_model():
    # let create a performance file under root folder
    output_file = f"{PROJECT_PATH}/Ticker_performance_arima.csv"
    # files headers: ["ticker", "interval", "price_type", "starting", "ending", "performance"]
    # to simplify, every time we retrain, simply overwrite.
    # add trained time at the end
    interval = '1d'
    price_type = 'Close'
    tickers = get_ticker_list()

    # we use try except here !!!
    # reason: some ticker may not exist; some possible implete data; etc.
    # we don't handle exception. simple pass
    # output list of tuples
    # use MAPE (mean absolute percentage error) to measure performance and rank
    performances = []
    for ticker in tickers:
        try:
            forecast, test_diffed,  order, original =\
                train_autoarima_for_batch(
                    ticker=ticker,
                    interval=interval,
                    price_type=price_type,
                    plot_percentage_change=False,
                    test_size=0.2,
                    predict_n = 5)

            p, d, q = order
            next_forecat = forecast.values[0]
            next_true = original.values[0]
            mape = np.mean([abs(b-a)/b for a, b in zip(forecast.values, original.values)])
            performances.append((ticker, interval, price_type, next_forecat, next_true, mape, p, d, q))

        except:
            pass
    df_result = pd.DataFrame(performances,
        columns = ["ticker", "interval", "price_type", "next_forecast", "next_true", "mape", "p", "d", "q"] )
    df_result["trained_on"] =datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M")

    # ranking
    df_result["rank"] = df_result["mape"].rank(ascending=True, method="first").astype(int)
    df_result.sort_values(by="rank", ascending=True, inplace=True)
    df_result.to_csv(output_file)

    # arima_model()
    # print('adf')