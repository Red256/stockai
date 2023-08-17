

import os
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import json
from bs4 import BeautifulSoup
import pandas as pd
import csv
import numpy as np

url_nyse_info = 'https://raw.githubusercontent.com/stempro/stockai/stockai/data/NYSE_ticker_info.csv'
url_nasdaq_info = 'https://raw.githubusercontent.com/stempro/stockai/stockai/data/NASDAQ_ticker_info.csv'
support_folder_data="/content/drive/MyDrive/stockai/data/MyStockAiProject"

class choose_ticker():
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
      df_nyse = pd.read_csv(url_nyse_info)
      df_nasdaq = pd.read_csv(url_nasdaq_info)
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

      ## 1: set the masterfile.
      your_project_file = f"{support_folder_data}/Project_ticker_list.csv"

      # if file exists and not to regenerate, simply return the master file
      if os.path.exists(your_project_file) and (not refresh_list):
          return pd.read_csv(your_project_file)

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
      df_.to_csv(your_project_file)

      return df_

class download_ticker():
  def track_progress(jsonobj=None, action="load"):
    """
        action: load, dump
        to simplify, we use all fix file name and schema.
    """
    track_json_file = f"{support_folder_data}/yf_progress.json"
    if action.lower() == "load":
        if os.path.exists(track_json_file):
            return json.load(open(track_json_file, 'r'))
        else:
            return {}
    else:
        if not jsonobj:
            return
        json.dump(jsonobj, open(track_json_file, 'w'))

  def get_histories(intervals=["1m","5m","1d"],
                  max_tickers_per_call = 100,
                  reload=True,
                  your_project_file = None):
    """
        action: load, dump
        to simplify, we use all fix file name and folder name.
        time periods 1d, 5d, 1y
        interval: 1m, 5m, 1d. for 1m and 5m, we take 5 days of data
            for 1d . with take 1 y
        filename convention: interval_period_yyyymmmddd.csv
        progress_json:
            1m: True
    """
    progress_json = track_progress(action="load")

    data_folder = f"{support_folder_data}"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if not your_project_file:
        your_project_file = f"{support_folder_data}/Project_ticker_list.csv"

    ticker_df = pd.read_csv(your_project_file)
    tickers = list(ticker_df['Ticker'])[:10]
    n_tickers = len(tickers)
    datetime_str = datetime.strftime(datetime.now(), "%YYYY%m%d%H%M")

    for interval in intervals:
        if (not reload) and interval in progress_json:
            continue
        period = "5d" if interval in ["1m", "5m"] else "1y"
        for i in range(0, n_tickers, max_tickers_per_call):

            csv_file = f"{data_folder}/{period}_{interval}_{datetime_str}_{i}.csv"

            tickers_ = tickers[i:(i+max_tickers_per_call)]

            df_ = yf.download(tickers_, interval=interval, period=period, threads=True,
                              prepost = False, repair = True)
            df_ = df_.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)
            df_ = df_.reset_index()
            if df_.shape[0]>0:
                df_.to_csv(csv_file)
                progress_json[interval] = "Downloaded"
                #save it right away
                track_progress(jsonobj=progress_json, action="save")

    return progress_json
