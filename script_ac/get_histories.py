
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import sys
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np

_SCRIPTS_FOLDER = "script_ac"
PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/{_SCRIPTS_FOLDER}/data"
PROJECT_TICKERS = f"{PROJECT_PATH}/Project_ticker_list.csv"

sys.path.append(PROJECT_PATH)
from script_ac.generate_ticker_list import choose_and_save_my_list

# get historical data for analysis
def download_histories(intervals=["1m","5m","1d"],
                  max_tickers_per_call = 100 ):
    """
        action: load, dump
        to simplify, we use all fix file name and folder name.
        time periods 1d, 5d, 1y
        interval: 1m, 5m, 1d. for 1m and 5m, we take 5 days of data
            for 1d. with take 1 y
        filename convention: interval_period_yyyymmmddd.csv

    """
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    # if the ticker file is not generated, generate a default list
    if not PROJECT_TICKERS:
        choose_and_save_my_list()

    ticker_df = pd.read_csv(PROJECT_TICKERS)

    # get ticker list and count of tickers
    tickers = list(set(ticker_df['Ticker']))
    n_tickers = len(tickers)

    datetime_str = datetime.strftime(datetime.now(), "%Y%m%d%H%M")

    for interval in intervals:
        period = "5d" if interval in ["1m", "5m"] else "1y"
        for i in range(0, n_tickers, max_tickers_per_call):

            csv_file = f"{DATA_PATH}/{period}_{interval}_{datetime_str}_{i}.csv"

            tickers_ = tickers[i:(i+max_tickers_per_call)]

            df_ = yf.download(tickers_, interval=interval, period=period, threads=True,
                              prepost = False, repair = True)
            df_ = df_.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)
            df_ = df_.reset_index()
            if df_.shape[0]>0:
                df_.to_csv(csv_file)

def get_one_ticker_df(ticker,interval ):
    """
        goals: get data for ticker "ticker
        input: ticker, e.g. AAPL, interval: 1d, 1m or 5m
        output: data frame
    """
    files = os.listdir(DATA_PATH)
    files = [f for f in files if re.search(interval, f)]
    if not files:
        return pd.DataFrame()

    df = pd.DataFrame()
    for file_ in files:
        df = pd.read_csv(f"{DATA_PATH}/{file_}")
        df = df[df["Ticker"]==ticker]
        if not df.empty:
            break
    return df
