
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import json
from bs4 import BeautifulSoup
import pandas as pd
import csv
import numpy as np
import re

_SCRIPTS_FOLDER = "scripts_rz"
PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/{_SCRIPTS_FOLDER}/data"

URL_NYSE = f"{PROJECT_PATH}/NYSE_ticker_info.csv"
URL_NASDAQ = f"{PROJECT_PATH}/NASDAQ_ticker_info.csv"
PROJECT_TICKERS = f"{DATA_PATH}/Project_ticker_list.csv"
# This function only called by choose_and_save_my_list

def choose_(industries,
            sectors,
            marketcap_from,
            marketcap_to,
            averageVolume_from,
            averageVolume_to,
            limit_n):
    """
        this function is called by choose_and_save_my_list
        parameters definition and default values are explain in that function
    """

    ## 1: read stock list and combine into one single file
    df_nyse = pd.read_csv(URL_NYSE)
    df_nasdaq = pd.read_csv(URL_NASDAQ)
    df_total = pd.concat([df_nyse, df_nasdaq]) # combine

    #make a copy of data (not address)
    df_ = df_total.copy()
    #advanced: df_['industry'].str.lower().isin([x.lower() for x in industries])

    ## start filtering by industries, sectors
    if industries:
        df_ = df_[df_['industry'].isin(industries)]
        if df_.shape[0]<=limit_n:
            return df_, df_total

    if sectors:
        df_ = df_[df_['sector'].isin(sectors)]
        if df_.shape[0]<=limit_n:
            return df_, df_total

    # filtering by marketcap and volume
    df_ = df_[df_['marketCap'].between(marketcap_from, marketcap_to)]
    if df_.shape[0]<=limit_n:
        return df_, df_total

    df_ = df_[df_['averageVolume'].between(averageVolume_from, averageVolume_to)]
    if df_.shape[0]<=limit_n:
        return df_, df_total
    else:
        return df_.sample(limit_n), df_total

def choose_and_save_my_list(extras=[],
                industries=[],
                sectors=[],
                marketcap_from=-np.inf,
                marketcap_to=np.inf,
                averageVolume_from=-np.inf,
                averageVolume_to=np.inf,
                limit_n = 50,
                refresh_list=False):
    """
        goal: generate a subset of stock symbols.
        using filter: industries, sections, market cap and volume.
        limit_n: set a limit of stocks
        if to regenerate the list, set refresh_list to True
    """

    ## 1: set the masterfile.
    # PROJECT_TICKERS

    # if file exists and not to regenerate, simply return the master file
    if os.path.exists(PROJECT_TICKERS) and (not refresh_list):
        return pd.read_csv(PROJECT_TICKERS)

    ## 2: call the choose_ function to filter the list of companies we want
    df_, df_total = choose_(industries=industries, sectors=sectors,
        marketcap_from=marketcap_from, marketcap_to=marketcap_to,
        averageVolume_from=averageVolume_from,
        averageVolume_to=averageVolume_to, limit_n = limit_n)

    ## 3. add extra companies that we are interested in
    if extras:
        extras = [ticker for ticker in extras if ticker not in list(df_["Ticker"])]
        df_ = pd.concat([df_, df_total[df_total["Ticker"].isin(extras)]])

    # save to YOUR google drive
    df_.to_csv(PROJECT_TICKERS)

def get_ticker_list():
    """
        goals: get the list of tickers from the file we generated
        input: the ticker file PROJECT_TICKERS
        output: ticker list
    """
    ### take short cut: take the tickers from historical data
    files = os.listdir(DATA_PATH)
    files = [f for f in files if re.search('1d', f)]

    tickers = []
    for file_ in files:
        df_ = pd.read_csv(f"{DATA_PATH}/{file_}")
        tickers_ = list(set(df_["Ticker"].to_list()))
        tickers.extend(tickers_)
    return sorted(set(tickers))
    # df = pd.read_csv(PROJECT_TICKERS)
    # return sorted(df['Ticker'].tolist())