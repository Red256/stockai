
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
About ticker_eda.py:
    this module holds all the functions for loading ticker stock data
    use help(ticker_eda) to list all functions
"""

_SCRIPTS_FOLDER = "scripts_rz"
PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/{_SCRIPTS_FOLDER}/data"

def get_one_ticker(ticker=None, interval='1m', recent_n = 0, data_path_=None):
    """
        goals: return one ticker dataframe
            if ticker is None, then pick one random
            if recent_n is None, then return every rows
        input: interval. choices: 1m, 5m,1d
            ticker
            recent_n
        output: df
    """
    if not data_path_:
        data_path_ = DATA_PATH

    files = os.listdir(data_path_)

    files = [data_path_+'/'+f for f in files if re.search(interval, f)]

    if not files:
        return

    if not ticker:
        i = np.random.randint(0,len(files))
        df_ = pd.read_csv(files[0])
        ticker = np.random.choice(df_["Ticker"])
        df_ = df_[df_["Ticker"]==ticker]
    else:
        df_ = pd.DataFrame()
        for f in files:
            df_ = pd.read_csv(f)
            df_ = df_[df_['Ticker']==(ticker.upper())]
            if df_.shape[0]>0:
                break
    if (not df_.empty) and recent_n>0:
        df_ = df_.iloc[:, -recent_n: ]

    return df_

def get_sma_data_one_ticker_(df, ticker,recent_n, base_price,
                         win_1, win_2, min_periods):
    """
        goals: filter and get the data for ticker with recent_n data points
        input: df from raw file
                base_price: e.g., Close, Open
                ticker, e.g., AAPL
                recent_n: e.g, 100
    """
    df_ = df.copy()
    df_ =df_[df_['Ticker']==(ticker.upper())]
    df_ = df_.iloc[-recent_n:]

    df_[f"{base_price}_{win_1}"] = df_[base_price].rolling(window=win_1,
                min_periods=min_periods).mean()
    df_[f"{base_price}_{win_2}"] = df_[base_price].rolling(window=win_2,
                min_periods=min_periods).mean()

    return df_

def get_ewm_data_one_ticker_(df, ticker,recent_n, base_price,  span_1, span_2):
    """
        goals: filter and get the data for ticker with recent_n data points
        input: df from raw file
                base_price: e.g., Close, Open
                ticker, e.g., AAPL
                recent_n: e.g, 100
    """
    df_ = df.copy()
    df_ =df_[df_['Ticker']==(ticker.upper())]
    df_ = df_.iloc[-recent_n:]

    df_[f"{base_price}_{span_1}"] = df_[base_price].ewm(span=span_1,adjust=False).mean()
    df_[f"{base_price}_{span_2}"] = df_[base_price].ewm(span=span_2,adjust=False).mean()

    return df_

def get_data_one_ticker_(df, ticker,recent_n, base_price, **kwargs):
    """
        goals: filter and get the data for ticker with recent_n data points
        input: df from raw file
                base_price: e.g., Close, Open
                ticker, e.g., AAPL
                recent_n: e.g, 100
                use kwargs.
                    expected parameters:
                        win_1, win_2 -- for rolling window. sma
                        span_1, span_2 -- for ewm
    """
    df_ = df.copy()
    df_ =df_[df_['Ticker']==(ticker.upper())]
    df_ = df_.iloc[-recent_n:]

    # for sma min_periods
    min_periods = kwargs.get('min_periods')

    for k, v in kwargs:
        if "span" in k.lower():
            df_[f"{base_price}_{k}_{v}"] = df_[base_price].ewm(span=v,adjust=False).mean()
        if "win" in k.lower():
            df_[f"{base_price}_{k}_{v}"] = df_[base_price].rolling(window=v).mean()
    return df_

def list_files_(data_path_):
    """
        goal: list files in data_path
        input: path for your data
        output list
    """
    files = [f"{data_path_}/{f}" for f in os.listdir(data_path_)]

    return files

def choose_a_file(interval='1m', data_path_=None):
    """
    goal: get a file that is of interval
    input: path for your data abd interval
    output: if exists such a file, output it. otherwise, renturn None
    """
    if not data_path_:
        data_path_ = DATA_PATH
    files = list_files_(data_path_)

    files = [f for f in files if re.search(interval, f)]

    if files:
        return files[0]
    return None

def data_for_visualize(one_file, tickers=None):
    """
        goals: get the data for visualization. if tickers in None, then list all,
            otherwise, list data for tickers.
        input: one file
        output: the dataframe as well as list of tickers
        note: tickers could be case sensitive
    """
    df_data = pd.read_csv(one_file)

    if tickers:
        if isinstance(tickers, str):
            tickers = [tickers]
            tickers = [t.upper() for t in tickers]
        df_data = df_data[df_data['Ticker'].isin(tickers)]

    # return actual list of data and tickers.
    return list(df_data['Ticker'].unique()), df_data

def plot_it_sma(ticker, interval, base_price,win_1, win_2, df_, N_ticks=40):
    """
        goals: plot one plot for one ticker
        input: df_ for the ticker.
        output: plots
        Note: show N ticks. so
    """
    plt.figure(figsize=(8, 4))
    price1 = base_price
    price2 = f"{base_price}_{win_1}"
    price3 = f"{base_price}_{win_2}"

    sns.lineplot(data=df_, x="Date", y=price1, label=price1)
    sns.lineplot(data=df_, x="Date", y=price2, label=price2)
    sns.lineplot(data=df_, x="Date", y=price3, label=price3)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f"{ticker}'s Simiple Moving Average (SMA) of {base_price} Price by {interval}", fontsize=8)
    plt.grid()

    every_n_ticks = int(df_.shape[0]/N_ticks)+1

    plt.xticks(rotation=90)
    plt.xticks(range(0, len(df_['Date']), every_n_ticks))

    plt.legend()
    plt.show()


def plot_it_ewm(ticker, interval, base_price,span_1, span_2, df_, N_ticks=40):
    """
        goals: plot one plot for one ticker
        input: df_ for the ticker.
        output: plots
        Note: show N ticks. so
    """
    plt.figure(figsize=(8, 4))
    price1 = base_price
    price2 = f"{base_price}_{span_1}"
    price3 = f"{base_price}_{span_2}"

    sns.lineplot(data=df_, x="Date", y=price1, label=price1)
    sns.lineplot(data=df_, x="Date", y=price2, label=price2)
    sns.lineplot(data=df_, x="Date", y=price3, label=price3)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f"{ticker}'s Exponential Weighted Moving Average {base_price} Price by {interval}", fontsize=8)
    plt.grid()

    every_n_ticks = int(df_.shape[0]/N_ticks)+1

    plt.xticks(rotation=90)
    plt.xticks(range(0, len(df_['Date']), every_n_ticks))

    plt.legend()
    plt.show()


def visualize_sma(base_price, tickers=None, win_1=50, win_2=100, recent_n=1000, min_periods=50,
              interval='1d', data_path_=None ):
    """
        goals: visualize recent_n data points for base_price with mas(moving average)
            if not tickers, visualize all tickers
        input: base_price, e.g., Close, Open etc.
        ouput, plots
    """
    one_file = choose_a_file(interval=interval, data_path_=data_path_)
    if not one_file:
        return

    tickers, df_data = data_for_visualize(one_file=one_file, tickers=tickers)

    if not tickers:
        return

    n_tickers = len(tickers)
    for ticker in tickers:
        df_ = get_sma_data_one_ticker_(df=df_data, ticker=ticker, recent_n=recent_n,
                                   base_price=base_price, win_1=win_1,
                                   win_2=win_2, min_periods=min_periods)
        return plot_it_sma(ticker,interval, base_price,win_1, win_2, df_)



def visualize_ewm(base_price, tickers=None, span_1=5, span_2=10, recent_n=1000,
              interval='1d', data_path_=None ):
    """
        goals: visualize recent_n data points for base_price with mas(moving average)
            if not tickers, visualize all tickers
        input: base_price, e.g., Close, Open etc.
        ouput, plots
    """
    one_file = choose_a_file(interval=interval, data_path_=data_path_)
    if not one_file:
        return

    tickers, df_data = data_for_visualize(one_file=one_file, tickers=tickers)

    if not tickers:
        return

    n_tickers = len(tickers)
    for ticker in tickers:
        df_ = get_ewm_data_one_ticker_(df=df_data, ticker=ticker, recent_n=recent_n,
                                   base_price=base_price, span_1=span_1,  span_2=span_2 )
        plot_it_ewm(ticker,interval, base_price,span_1, span_2, df_)
        return

################################ modify for streamlit ################################
# added for one ticker to show in Streamlit
# sma
def visualize_sma_one_ticker(base_price, ticker, win_1=50, win_2=100, recent_n=1000, min_periods=50,
              interval='1d', data_path_=None ):
    """
        goals: visualize recent_n data points for base_price with mas(moving average)
            for One Ticker Only
        input: base_price, e.g., Close, Open etc.
        ouput, plots
    """
    one_file = choose_a_file(interval=interval, data_path_=data_path_)
    if not one_file:
        return

    _, df_data = data_for_visualize(one_file=one_file, tickers=[ticker])

    if df_data.empty:
        return pd.DataFrame()


    df_ = get_sma_data_one_ticker_(df=df_data, ticker=ticker, recent_n=recent_n,
                                base_price=base_price, win_1=win_1,
                                win_2=win_2, min_periods=min_periods)
    plot_it_sma(ticker,interval, base_price,win_1, win_2, df_)

#wma
def visualize_ewm_one_ticker(base_price, ticker=None, span_1=5, span_2=10, recent_n=1000,
              interval='1d', data_path_=None ):
    """
        goals: visualize recent_n data points for base_price with mas(moving average)
            if not tickers, visualize all tickers
        input: base_price, e.g., Close, Open etc.
        ouput, plots
    """
    one_file = choose_a_file(interval=interval, data_path_=data_path_)
    if not one_file:
        return

    _, df_data = data_for_visualize(one_file=one_file, tickers=[ticker])

    if df_data.empty:
        return pd.DataFrame()

    df_ = get_ewm_data_one_ticker_(df=df_data, ticker=ticker, recent_n=recent_n,
                                base_price=base_price, span_1=span_1,  span_2=span_2 )
    plot_it_ewm(ticker,interval, base_price,span_1, span_2, df_)
