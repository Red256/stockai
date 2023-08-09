
import pandas as pd
import os
import sys

PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/data"
ACCOUNT_PATH = f"{PROJECT_PATH}/accounts"
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_template"
sys.path.append(PROJECT_PATH)

from scripts_template.generate_ticker_list import choose_and_save_my_list, get_ticker_list
from scripts_template.get_histories import download_histories, get_one_ticker_df


def get_balance():
    return 20099.69, 18135.75

def get_positions():
    return pd.read_csv(f"{ACCOUNT_PATH}/positions.csv")

def get_pending_orders():
    columns = ['ticker','order number', 'action', 'model', 'order date(utc)', 'ordertype', 'shares', 'amount', 'rsi_value', 'price']
    return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    # return pd.DataFrame({"ticker": ["AAPL"], "action": ["buy"], "model": ["ARIMA"], "order date(utc)": ["2023-08-08 12:23"], "ordertype": ["GTC"],
    #                      "shares": [150]}),  pd.DataFrame({"ticker": ["AAPL"], "action": ["sell"], "model": ["ARIMA"], "order date(utc)": ["2023-08-08 12:23"], "ordertype": ["GTC"],
    #                      "shares": [150]})
# df1, df2 = get_pending_orders()
# print(22)