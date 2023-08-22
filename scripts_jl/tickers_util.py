
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

class choose_ticker():
  def choose_(industries,
              sectors,
              marketcap_from,
              marketcap_to,
              averageVolume_from,
              averageVolume_to,
              limit_n):
      df_nyse = pd.read_csv(url_nyse_info)
      df_nasdaq = pd.read_csv(url_nasdaq_info)
      df_total = pd.concat([df_nyse, df_nasdaq]) # combine

      df_ = df_total.copy()

      if industries:
          df_ = df_[df_['industry'].isin(industries)]
          if df_.shape[0]<=limit_n:
              return df_, df_total

      if sectors:
          df_ = df_[df_['sector'].isin(sectors)]
          if df_.shape[0]<=limit_n:
              return df_, df_total

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

      your_project_file = f"{support_folder_master}/Project_ticker_list.csv"

      if os.path.exists(your_project_file) and (not refresh_list):
          return pd.read_csv(your_project_file)

      df_, df_total = choose_(industries=industries, sectors=sectors,
          marketcap_from=marketcap_from, marketcap_to=marketcap_to,
          averageVolume_from=averageVolume_from,
          averageVolume_to=averageVolume_to, limit_n = limit_n)

      if extras:
          extras = [ticker for ticker in extras if ticker not in list(df_["Ticker"])]
          df_ = pd.concat([df_, df_total[df_total["Ticker"].isin(extras)]])

      df_.to_csv(your_project_file)

      return df_
