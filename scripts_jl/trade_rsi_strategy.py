
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/data"
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_jl"
sys.path.append(PROJECT_PATH)

from scripts_jl.generate_ticker_list import choose_and_save_my_list, get_ticker_list
from scripts_jl.get_histories import download_histories, get_one_ticker_df
from scripts_jl.rs_rsi import compute_RSI
from utility import DataProcessing

"""
    scripts are tailored for streamlit visualization
    after get the df, we take these steps:
    step 1: use DataProcessing from utility to "clean" the data
    step 2: call compute_RSI from rs_rsi module to the dataframe
    stpe 3: run implement_rsi_strategy to get the pricing and signals
    step 4: plot positions
"""

def implement_rsi_strategy(ticker, interval, price_type="Close", rsi_name='RSI_14'):
    """
        goals: generate a price point for trading
        input: df_ dataframe. e.g., our df_enrich.
                price_type:
                start with signal = 0: means the first signal has to be buy
                signal: 1 ---- buy
                        -1 ---- sell
    """
    # get df_ by calling get_one_ticker_df
    df_ = get_one_ticker_df(ticker=ticker, interval=interval)

    # "clean it"
    dp = DataProcessing() # the only time we use Class
    df_ = dp.cleaning(df_)

    # compute rsi
    df_ = compute_RSI(df_)

    buy_price = [np.nan]
    sell_price = [np.nan]
    rsi_signal = [0]

    signal = 0 #starts with noghtin.

    rsi = df_[rsi_name].tolist()
    prices = df_[price_type].tolist()

    for i in range(1, len(rsi)):
        if rsi[i-1] > 30 and rsi[i] < 30: # BUY!!
            if signal == 0:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
        elif rsi[i-1] < 70 and rsi[i] > 70: ### SELL
            if signal > 0:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                rsi_signal.append(-1)
                signal = 0
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            rsi_signal.append(0)

    return buy_price, sell_price, rsi_signal, df_


########### plot_trading_points will be used in streamlit plotting  ###########
def plot_trading_points(ticker,
                        interval,
                        price_type='Close',
                        rsi_name='RSI_14',
                        N_ticks=40):
    """
        goals: visualize and validate about trading points
        input: df_, e.g., df_enrich
               buy_price and sell_price. these two are calculated from trading
                algorithm
    """
    # call implent_rsi_strategy to get buy/sell prices
    buy_price, sell_price, rsi_signal, df_ = \
        implement_rsi_strategy(ticker=ticker, interval=interval, \
                price_type=price_type, rsi_name=rsi_name)

    df_enrich = df_.copy()
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 4))
    fig.suptitle(f'{ticker.upper()} {price_type} Price, RSI and Buy&Sell', fontsize=4)

    #ax1.plot(data=df_, x="Date", y="Close", color='blue')
    sns.lineplot(data=df_enrich, x="Date", y=price_type, label=price_type, ax=ax1)
    every_n_ticks = int(df_enrich.shape[0] / N_ticks) + 1
    ax1.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax1.set_xticklabels(df_['Date'][::every_n_ticks], rotation=90)

    ax1.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax1.set_xticklabels([])  # Hide the x-axis labels
    ax1.grid(axis='x')
    ax1.set_xlabel('')

    ax1.plot(df_enrich['Date'], buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
    ax1.plot(df_enrich['Date'], sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')


    # ax2.plot(df_[f'RSI_{span}'], label=f'{span}-day RSI', color='orange')
    ax2.plot(df_enrich['Date'], df_enrich[rsi_name], label=f'{rsi_name}-day RSI', color='orange')
    ax2.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax2.set_xticklabels(df_['Date'][::every_n_ticks], rotation=90)

    lines = {20:"red", 30:"orange", 50:"blue", 70:"yellowgreen", 80:"green"}
    ax2.axhline(0, linestyle='--', alpha=0.1, color='black')
    ax2.axhline(100, linestyle='--', alpha=0.1, color='black')
    for val, color in lines.items():
        ax2.axhline(val, linestyle='--', alpha=0.5, color=color)
        plt.text(0.5, val, f"{val}%")

    ax1.set_title(f'{ticker.upper()} {price_type} Price and Buy/Sell')
    ax2.set_title(f'{ticker.upper()} RSI')

    ax1.grid(axis='x')
    ax2.grid(axis='x')

    plt.xticks(rotation=90)
    plt.show()

def create_position(ticker, interval, rsi_signal, price_type='Close', rsi_name='RSI_14', starting_fund=100000 ):
    """
        create positions use the signals from trading strategy
        input: df_ is the pricing history
               rsi_signal: as if we bought or sold the stock
    """
    # call implent_rsi_strategy to get buy/sell prices
    _, _, rsi_signal, df_ = \
        implement_rsi_strategy(ticker=ticker, interval=interval, \
                               price_type=price_type, rsi_name=rsi_name)

    df_position = df_.copy()
    df_position["Position_Signal"] = rsi_signal

    shares = []
    vals = []
    share  = 0

    for i, row in df_position.iterrows():
        transaction_price, signal = row['Close'], row['Position_Signal']
        if signal == 1: #buy
            share = starting_fund/transaction_price
        elif signal == -1:
            starting_fund = share * transaction_price
            share = 0

        shares.append(share)
        vals.append(starting_fund )

    df_position['Shares'] = shares
    df_position['Value'] = vals
    return df_position


def plot_position(df_,
                ticker,
                buy_price,
                sell_price,
                price_type='Close',
                rsi_name='RSI_14',
                N_ticks=40):
    """
        goals: visualize our positions
        input: df_, e.g., df_position
    """

    # call create_position(ticker, interval, rsi_signal, price_type='Close', rsi_name='RSI_14', starting_fund=100000 ):
    df_ = create_position(ticker=ticker, interval=interval)
    df_position = df_.copy()
    plt.figure(figsize=(8,4))

    ax1 = sns.lineplot(data=df_position, x="Date", y=price_type, label=price_type )
    every_n_ticks = int(df_position.shape[0] / N_ticks) + 1
    ax1.set_xticks(range(0, len(df_position['Date']), every_n_ticks))
    ax1.set_xticklabels(df_['Date'][::every_n_ticks], rotation=90)
    ax1.set_xticks(range(0, len(df_position['Date']), every_n_ticks))
    ax1.plot(df_position['Date'], buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
    ax1.plot(df_position['Date'], sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
    min_price,max_price = df_position[price_type].min()*0.85, df_position[price_type].max()*1.03

    ax1.set_ylim((min_price,max_price ))
    plt.legend(loc='lower left')

    ax2 = ax1.twinx()
    sns.lineplot(data=df_position, x="Date", y="Value", label="Position", color='green',linestyle='--', ax=ax2)
    min_val, max_val = df_position['Value'].min()*0.85, df_position['Value'].max()*1.03
    ax2.set_ylim((min_val,max_val ))


    plt.title(f'{ticker.upper()} {price_type} Price, RSI and Position')

    ax2.grid(None)
    ax2.grid(axis='x')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.show()


############# add create and plot to streamlit #############
def create_plot_position(ticker, interval, \
        price_type='Close', rsi_name='RSI_14', starting_fund=100000, N_ticks=40 ):
    """
        create positions use the signals from trading strategy
        input: df_ is the pricing history
               rsi_signal: as if we bought or sold the stock
    """
    # call implent_rsi_strategy to get buy/sell prices
    buy_price, sell_price, rsi_signal, df_ = \
        implement_rsi_strategy(ticker=ticker, interval=interval, \
                               price_type=price_type, rsi_name=rsi_name)

    df_position = df_.copy()
    df_position["Position_Signal"] = rsi_signal

    shares = []
    vals = []
    share  = 0

    for i, row in df_position.iterrows():
        transaction_price, signal = row['Close'], row['Position_Signal']
        if signal == 1: #buy
            share = starting_fund/transaction_price
        elif signal == -1:
            starting_fund = share * transaction_price
            share = 0

        shares.append(share)
        vals.append(starting_fund )

    df_position['Shares'] = shares
    df_position['Value'] = vals

    ####### ploting
    plt.figure(figsize=(8,4))

    ax1 = sns.lineplot(data=df_position, x="Date", y=price_type, label=price_type )
    every_n_ticks = int(df_position.shape[0] / N_ticks) + 1
    ax1.set_xticks(range(0, len(df_position['Date']), every_n_ticks))
    ax1.set_xticklabels(df_['Date'][::every_n_ticks], rotation=90)
    ax1.set_xticks(range(0, len(df_position['Date']), every_n_ticks))
    ax1.plot(df_position['Date'], buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
    ax1.plot(df_position['Date'], sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
    min_price,max_price = df_position[price_type].min()*0.85, df_position[price_type].max()*1.03

    ax1.set_ylim((min_price,max_price ))
    plt.legend(loc='lower left')

    ax2 = ax1.twinx()
    sns.lineplot(data=df_position, x="Date", y="Value", label="Position", color='green',linestyle='--', ax=ax2)
    min_val, max_val = df_position['Value'].min()*0.85, df_position['Value'].max()*1.03
    ax2.set_ylim((min_val,max_val ))


    plt.title(f'{ticker.upper()} {price_type} Price, RSI and Position')

    ax2.grid(None)
    ax2.grid(axis='x')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.show()


