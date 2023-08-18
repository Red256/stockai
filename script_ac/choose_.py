import pandas as pd
import numpy as np
import os

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
    df_nyse = pd.read_csv("https://raw.githubusercontent.com/stempro/StocksFolder/StocksFolder/data/NYSE_ticker_info.csv")
    df_nasdaq = pd.read_csv("https://raw.githubusercontent.com/stempro/StocksFolder/StocksFolder/data/NASDAQ_ticker_info.csv")
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
        set a limit of stocks

        if to regenerate the list, set refresh_list to True
    """

    ## 1: set the masterfile.
    your_project_file = "/content/drive/MyDrive/StocksFolder/Project_ticker_list.csv"

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