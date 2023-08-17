

import os
import sys
import re
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from importlib import reload
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
import random

#Plotting and
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#Scikit-Learn for Modeling
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error,mean_squared_log_error


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
# Checking Stasionarity - Dicky Fuller Test
from statsmodels.tsa.stattools import adfuller
from matplotlib.ticker import MaxNLocator

from pmdarima.arima import auto_arima

_SCRIPTS_FOLDER = "scripts_rz"
PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/{_SCRIPTS_FOLDER}/data"
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_rz"
sys.path.append(PROJECT_PATH)

from scripts_rz.generate_ticker_list import choose_and_save_my_list, get_ticker_list
from scripts_rz.get_histories import download_histories, get_one_ticker_df

# plot a stock's OHLC chart
def plot_stock(ticker, interval, N_ticks=40):
    """
    goals: visualize stock prices
    output: chart
    note: for streamlit
    Args:
        ticker (_type_): _description_
        interval (_type_): _description_
        N_ticks (int, optional): _description_. Defaults to 40.
    """
    df_ = get_one_ticker_df(ticker=ticker, interval=interval)
    plt.figure(figsize=(8, 4))

    sns.lineplot(data=df_, x="Date", y='Open', label='Open')
    sns.lineplot(data=df_, x="Date", y='Close', label='Close')
    sns.lineplot(data=df_, x="Date", y='High', label='High')
    sns.lineplot(data=df_, x="Date", y='Low', label='Low')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f"{ticker}'s OHLC", fontsize=8)
    plt.grid()

    every_n_ticks = int(df_.shape[0]/N_ticks)+1

    plt.xticks(rotation=90)
    plt.xticks(range(0, len(df_['Date']), every_n_ticks))

    plt.legend()
    plt.show()


def test_stationarity(ticker, interval, price_type, N_ticks=30, plot_percentage_change=True):
    """_summary_
    goal: learn about stationarity of a time series
    output: chart
    note: for streamlit
    Args:
        ticker (_type_): _description_
        interval (_type_): _description_
        price_type (_type_): _description_
        N_ticks (int, optional): _description_. Defaults to 30.
        plot_percentage_change (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    df_ = get_one_ticker_df(ticker=ticker, interval=interval)
    df_ = df_.set_index('Date')
    df_.dropna(inplace=True)

    rcParams['figure.figsize'] = 14,8

    plot_col = price_type
    if plot_percentage_change:
        df_[f'{price_type}_Per'] = df_[price_type].pct_change(1)*100
        plot_col = f'{price_type}_Per'

    timeseries = df_[plot_col]
    #Determining rolling statistics
    rolmean = timeseries.rolling(4).mean() # around 4 weeks on each month
    rolstd = timeseries.rolling(4).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')


    # Set x-axis tick positions and labels
    every_n_ticks = int(timeseries.shape[0]/N_ticks)+1
    plt.xticks(rotation=90)
    step_size = len(timeseries) // every_n_ticks
    tick_positions = range(0, len(timeseries), every_n_ticks)
    tick_labels = timeseries.index[tick_positions]  # Assuming 'timeseries' has a DatetimeIndex

    # Apply the tick positions and labels to the plot
    plt.xticks(tick_positions, tick_labels)

    plt.grid()

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')

    timeseries = timeseries.interpolate()
    timeseries = timeseries.fillna(method='bfill')

    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)

        if dfoutput['p-value'] < 0.05:
            print('result : time series is stationary')
        else : print('result : time series is not stationary')

    return dftest, dfoutput

#########################################################################

# # Checking Trend and Seasonality
def check_trend_seasonality(ticker, interval, price_type, N_ticks=30, plot_percentage_change=True):
    """_summary_
    goal: standard way of check stationarity, trend, cyclical of a time series
    output: chart
    note: for streamlit
    Args:
        ticker (_type_): _description_
        interval (_type_): _description_
        price_type (_type_): _description_
        N_ticks (int, optional): _description_. Defaults to 30.
        plot_percentage_change (bool, optional): _description_. Defaults to True.
    """
    df_ = get_one_ticker_df(ticker=ticker, interval=interval)
    df_ = df_.set_index('Date')
    df_.dropna(inplace=True)

    plot_col = price_type
    if plot_percentage_change:
        df_[f'{price_type}_Per'] = df_[price_type].pct_change(1)*100
        plot_col = f'{price_type}_Per'

    #decomposition = seasonal_decompose(df['Close'], freq=30)
    df_[plot_col] = df_[plot_col].fillna(method='bfill')
    decomposition = seasonal_decompose(df_[plot_col], model='additive', period=30)
    # additive: 'additive' means that the observed time series is considered to
    # be a sum of its trend, seasonality, and residual components.
    # another option is multiplicative
    #period: represents the period or length of the seasonality

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.subplot(411)
    plt.plot(df_[plot_col], label='Original')
    plt.legend(loc='best')
    plt.xticks([])
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.xticks([])
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.xticks([])
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    # Show 30 ticks on the x-axis for the last plot
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=30))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def show_train_test(ticker, interval, price_type, N_ticks=30, plot_percentage_change=True, test_size=0.2):
    """_summary_
    goal: show splitted train/test data
    output: chart for visualization
    note: not used in streamlit.
    Args:
        ticker (_type_): _description_
        interval (_type_): _description_
        price_type (_type_): _description_
        N_ticks (int, optional): _description_. Defaults to 30.
        plot_percentage_change (bool, optional): _description_. Defaults to True.
        test_size (float, optional): _description_. Defaults to 0.2.
    """
    df_ = get_one_ticker_df(ticker=ticker, interval=interval)
    df_ = df_.set_index('Date')
    df_.dropna(inplace=True)

    plot_col = price_type
    if plot_percentage_change:
        df_[f'{price_type}_Per'] = df_[price_type].pct_change(1)*100
        plot_col = f'{price_type}_Per'

    df_ = df_[[f"{price_type}_Per"]]

    #decomposition = seasonal_decompose(df['Close'], freq=30)
    df_[plot_col] = df_[plot_col].fillna(method='bfill')

    train, test = df_[3:int(len(df_)*(1- test_size))], df_[int(len(df_)*(1-test_size)):]

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(train, 'green', label='Train data')
    plt.plot(test, 'blue', label='Test data')
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=30))
    plt.xticks(rotation=90)

    plt.legend()
    plt.show()


def train_autoarima(ticker,
                     interval,
                     price_type,
                     plot_percentage_change=True,
                     test_size=0.2,
                     predict_n = 5):
    """
    goal: use autoarima identify p, d and q.
          use the p/d/q to retrain the time series
    output: charts and analysis
    note: for streamlit

    Args:
        ticker (_type_): _description_
        interval (_type_): _description_
        price_type (_type_): _description_
        plot_percentage_change (bool, optional): _description_. Defaults to True.
        test_size (float, optional): _description_. Defaults to 0.2.
        predict_n (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    df_ = get_one_ticker_df(ticker=ticker, interval=interval)
    df_ = df_.set_index('Date')
    df_.dropna(inplace=True)
    plot_col = price_type
    if plot_percentage_change:
        df_[f'{price_type}_Per'] = df_[price_type].pct_change(1)*100
        plot_col = f'{price_type}_Per'

    ### step 1: use auto arima to get p, d, q  and series summaries
    df_[plot_col] = df_[plot_col].fillna(method='bfill')

    series_ = df_[plot_col]
    train, test = series_[0:int(len(df_)*(1- test_size))], series_[int(len(df_)*(1-test_size)):]

    model_autoARIMA = auto_arima(train, start_p=0, start_q=0,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    #print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))

    ## get the orders
    order = model_autoARIMA.order
    p,d,q = order

    # now use ARIMA model to fit
    ######################## forecast use ARIMA. Feed with stationary data
    series_  = df_[plot_col]
    while d > 0:
        #series_ = np.diff(series_, n=1)
        series_ = series_.diff(1)
        d -= 1

    # split again train arima model with order. if d>0, both train and test data have been differenced
    index_train_end = int(len(df_)*(1- test_size))-1

    train, test = series_[0:index_train_end], series_[index_train_end:index_train_end+predict_n]
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast_values = fitted.get_forecast(steps=predict_n, alpha=0.05)

    fc_mean = forecast_values.predicted_mean
    n_len = len(test.index)
    lower_series = forecast_values.conf_int()[f'lower {plot_col}'][:n_len]
    upper_series = forecast_values.conf_int()[f'upper {plot_col}'][:n_len]

    n_len = len(test.index)
    fc_series_2 = fc_mean[:n_len]
    fc_series_2.index = test.index
    lower_series.index = test.index
    upper_series.index = test.index

    # Plot
    plt.figure(figsize=(10,5), dpi=100)
    plt.plot(train , label='training data')
    plt.plot(test , color = 'blue', label='Actual Stock Price')
    plt.plot(fc_series_2, color = 'orange',label='Predicted Stock Price')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k' )
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=30))
    plt.xticks(rotation=90)
    plt.show()

    # restore. only need to restore d
    original_series =  df_[plot_col][index_train_end:index_train_end+predict_n]
    forecast_diff_diff = fc_mean
    _, d, _ = order
    forecast = original_series
    while d>0:
        # First step of inverting the differencing: Undo the first differencing
        forecast_diff = np.cumsum(forecast_diff_diff) + original_series[-d] - original_series[-d-1]
        d -= 1
        # Second step of inverting the differencing: Undo the second differencing
        forecast = np.cumsum(forecast_diff) + original_series[-d]
    return forecast, model_autoARIMA.summary(), order


def train_autoarima_for_batch(ticker,
                     interval,
                     price_type,
                     plot_percentage_change=True,
                     test_size=0.2,
                     predict_n = 5):
    """
    goal: use autoarima identify p, d and q.
          use the p/d/q to retrain the time series
    output: stock price prediction
    note: for batch training
    return forecast, test_diffed,  order, original
    Args:
        ticker (_type_): _description_
        interval (_type_): _description_
        price_type (_type_): _description_
        plot_percentage_change (bool, optional): _description_. Defaults to True.
        test_size (float, optional): _description_. Defaults to 0.2.
        predict_n (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    df_ = get_one_ticker_df(ticker=ticker, interval=interval)
    df_ = df_.set_index('Date')
    df_.dropna(inplace=True)
    plot_col = price_type
    if plot_percentage_change:
        df_[f'{price_type}_Per'] = df_[price_type].pct_change(1)*100
        plot_col = f'{price_type}_Per'

    ### step 1: use auto arima to get p, d, q  and series summaries
    df_[plot_col] = df_[plot_col].fillna(method='bfill')

    series_ = df_[plot_col]
    train, test = series_[0:int(len(df_)*(1- test_size))], series_[int(len(df_)*(1-test_size)):]

    model_autoARIMA = auto_arima(train, start_p=0, start_q=0,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    #print(model_autoARIMA.summary())
    #model_autoARIMA.plot_diagnostics(figsize=(15,8))

    ## get the orders
    order = model_autoARIMA.order
    p,d,q = order

    # now use ARIMA model to fit
    ######################## forecast use ARIMA. Feed with stationary data
    series_  = df_[plot_col]
    while d > 0:
        #series_ = np.diff(series_, n=1)
        series_ = series_.diff(1)
        d -= 1

    # split again train arima model with order. if d>0, both train and test data have been differenced
    index_train_end = int(len(df_)*(1- test_size))-1

    train, test = series_[0:index_train_end], series_[index_train_end:index_train_end+predict_n]
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast_values = fitted.get_forecast(steps=predict_n, alpha=0.05)

    fc_mean = forecast_values.predicted_mean
    n_len = len(test.index)
    lower_series = forecast_values.conf_int()[f'lower {plot_col}'][:n_len]
    upper_series = forecast_values.conf_int()[f'upper {plot_col}'][:n_len]

    n_len = len(test.index)
    fc_series_2 = fc_mean[:n_len]
    fc_series_2.index = test.index
    lower_series.index = test.index
    upper_series.index = test.index

    # restore. only need to restore d
    original_series =  df_[plot_col][index_train_end:index_train_end+predict_n]
    forecast_diff_diff = fc_mean
    _, d, _ = order
    forecast = original_series
    while d>0:
        # First step of inverting the differencing: Undo the first differencing
        forecast_diff = np.cumsum(forecast_diff_diff) + original_series[-d] - original_series[-d-1]
        d -= 1
        # Second step of inverting the differencing: Undo the second differencing
        forecast = np.cumsum(forecast_diff) + original_series[-d]
    return forecast, test[:predict_n],  order, df_[plot_col][index_train_end: index_train_end+predict_n]
