
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/data"
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_template"
sys.path.append(PROJECT_PATH)

from scripts_jl.generate_ticker_list import choose_and_save_my_list, get_ticker_list
from scripts_jl.get_histories import download_histories, get_one_ticker_df


def compute_RSI(df_,
                      price_type='Close' ,
                      smoothening_type='ewm',
                      span=14,
                      min_periods=3):
    """
        goals: calculate RSI on closing price.
        input: pd series of closing data.
                smoothening_type: ewm or sma
                span: used to define decay.
                decay means: a process of reducing an amount by a
                            consistent percentage rate over a period of time
                time_window: min periods for ewm.
    """
    df_enrich = df_.copy()

    price_data = df_enrich[price_type]
    diff = price_data.diff(1).dropna()
    up_chg = 0 * diff #arrays
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[ diff>0 ]
    down_chg[diff < 0] = diff[ diff < 0 ]

    if smoothening_type=='ewm':
        up_series_avg   = up_chg.ewm(span=span , min_periods=min_periods).mean()
        down_series_avg = down_chg.ewm(span=span , min_periods=min_periods).mean()
    else: #sma
        up_series_avg   = up_chg.rolling(window=span , min_periods=min_periods).mean()
        down_series_avg = down_chg.rolling(window=span , min_periods=min_periods).mean()

    rs = abs(up_series_avg/down_series_avg)
    rsi = 100 - 100/(1+rs)

    df_enrich['Up'] =up_series_avg
    df_enrich['Down'] =down_series_avg
    df_enrich[f'RS_{span}'] =rs
    df_enrich[f'RSI_{span}'] =rsi

    return df_enrich

def plot_RSI(df_, ticker,
             span=14,
             min_periods=3,
             smoothening_type='ewm',
             price_type='Close',
             N_ticks=40):
    """
        goals: plot RSI.
        inputs: data frame for one stock.
                ticker: stock name
                price_type: Close price
                smoothening_type: ewm or sma (rolling)
                span: ewm decay
                N_ticks: for x-axis tick labeling
                min_periods: this is for cal ewm valuess. min periods to start
                    calculating
    """
    df_enrich = compute_RSI(df_=df_,
                      price_type=price_type ,
                      smoothening_type=smoothening_type,
                      span=span,
                      min_periods=min_periods)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 4))
    fig.suptitle(f'{ticker.upper()} {price_type} Price and RSI')

    #ax1.plot(data=df_, x="Date", y="Close", color='blue')
    sns.lineplot(data=df_enrich, x="Date", y=price_type, label=price_type, ax=ax1)
    every_n_ticks = int(df_enrich.shape[0] / N_ticks) + 1
    ax1.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax1.set_xticklabels(df_['Date'][::every_n_ticks], rotation=90)

    ax1.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax1.set_xticklabels([])  # Hide the x-axis labels
    ax1.grid(axis='x')
    ax1.set_xlabel('')


    # ax2.plot(df_[f'RSI_{span}'], label=f'{span}-day RSI', color='orange')
    ax2.plot(df_enrich['Date'], df_enrich[f'RSI_{span}'], label=f'{span}-day RSI', color='orange')
    ax2.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax2.set_xticklabels(df_['Date'][::every_n_ticks], rotation=90)

    lines = {20:"red", 30:"orange", 50:"blue", 70:"yellowgreen", 80:"green"}
    ax2.axhline(0, linestyle='--', alpha=0.1, color='black')
    ax2.axhline(100, linestyle='--', alpha=0.1, color='black')
    for val, color in lines.items():
        ax2.axhline(val, linestyle='--', alpha=0.5, color=color)
        plt.text(0.5, val, f"{val}%")

    ax1.set_title(f'{ticker.upper()} {price_type} Price')
    ax2.set_title(f'{ticker.upper()} RSI')

    ax2.grid(axis='x')

    plt.xticks(rotation=90)
    plt.show()

####################### make a function for displaying in streamlit #############################

def plot_RSI_streamlit(ticker,
             interval,
             span=14,
             min_periods=3,
             smoothening_type='ewm',
             price_type='Close',
             N_ticks=40):
    """
        goals: plot RSI.
        inputs: data frame for one stock.
                ticker: stock name
                price_type: Close price
                smoothening_type: ewm or sma (rolling)
                span: ewm decay
                N_ticks: for x-axis tick labeling
                min_periods: this is for cal ewm valuess. min periods to start
                    calculating
    """
    df_ =  get_one_ticker_df(ticker,interval) # from another module: download_histories
    if df_.shape[0] > 250:
        df_ = df_.iloc[-250:]

    df_enrich = compute_RSI(df_=df_,
                      price_type=price_type ,
                      smoothening_type=smoothening_type,
                      span=span,
                      min_periods=min_periods)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 4))
    fig.suptitle(f'{ticker.upper()} {price_type} Price and RSI', fontsize=8)

    #ax1.plot(data=df_, x="Date", y="Close", color='blue')
    sns.lineplot(data=df_enrich, x="Date", y=price_type, label=price_type, ax=ax1)
    every_n_ticks = int(df_enrich.shape[0] / N_ticks) + 1
    ax1.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax1.set_xticklabels(df_['Date'][::every_n_ticks], rotation=90)

    ax1.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax1.set_xticklabels([])  # Hide the x-axis labels
    ax1.grid(axis='x')
    ax1.set_xlabel('')


    # ax2.plot(df_[f'RSI_{span}'], label=f'{span}-day RSI', color='orange')
    ax2.plot(df_enrich['Date'], df_enrich[f'RSI_{span}'], label=f'{span}-day RSI', color='orange')
    ax2.set_xticks(range(0, len(df_enrich['Date']), every_n_ticks))
    ax2.set_xticklabels(df_['Date'][::every_n_ticks], rotation=90)

    lines = {20:"red", 30:"orange", 50:"blue", 70:"yellowgreen", 80:"green"}
    ax2.axhline(0, linestyle='--', alpha=0.1, color='black')
    ax2.axhline(100, linestyle='--', alpha=0.1, color='black')
    for val, color in lines.items():
        ax2.axhline(val, linestyle='--', alpha=0.5, color=color)
        plt.text(0.5, val, f"{val}%")

    ax1.set_title(f'{ticker.upper()} {price_type} Price')
    ax2.set_title(f'{ticker.upper()} RSI')

    ax2.grid(axis='x')

    plt.xticks(rotation=90)
    plt.show()


    # df = get_one_ticker_df('NWL', '1d')
    # plot_RSI(df_ = df, ticker='NWL')
    # print('abc')
