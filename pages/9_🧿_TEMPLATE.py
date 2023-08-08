import streamlit as st
import pandas as pd
import json
import re
import os
import sys
from datetime import datetime, timedelta
import alpaca_trade_api as alpaca
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

st.set_page_config(page_title= "Camp 2: In Action", page_icon = "🔢", layout="wide")

# data source: stockai/data
#scripts folder: stockai/scripts_xx. here stockai/scripts_template

PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/data"
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_template"
sys.path.append(PROJECT_PATH)

from scripts_template.generate_ticker_list import choose_and_save_my_list, get_ticker_list
from scripts_template.get_histories import get_histories, get_one_ticker_df

from scripts_template.ticker_eda import visualize_sma_one_ticker, visualize_ewm_one_ticker
from scripts_template.rs_rsi import plot_RSI, plot_RSI_streamlit

from scripts_template.trade_rsi_strategy import plot_trading_points, create_plot_position

from scripts_template.auto_arima import (
        plot_stock,
        test_stationarity,
        check_trend_seasonality,
        show_train_test,
        train_autoarima)
############################# Pay layout ################################################
st.markdown("""
    <style>
            .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-right: 1rem;
            padding-left: 1rem;
            }
    </style>
""", unsafe_allow_html=True)
############################# path for our custom libraries #############################
# padding_top,padding_left = 100, 1
# st.markdown(f'''
#            <style>
#             .appview-container .main .block-container{{
#                     padding-top: {padding_top}rem;
#                     padding-left: {padding_left}rem;   }}
#             </style>
#             ''', unsafe_allow_html=True,
# )
st.title(f"🔢 Template's StockAI Project")

######################################################## tabs ########################################################
listTabs =["🧑‍🏭Data Exploration",
           "🧑‍🎓Trade Strategies RSI",
           "🧑‍🎓Trade Strategies ARIMA",
           "📈 Model Training",
           "🔢 My Finished Product", "        "]

whitespace = 9
## Fills and centers each tab label with em-spaces
tabs = st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

# Shared data among all tabs:
stock_tickers = get_ticker_list()
intervals = ['1d', '1m', '5m']
prices = ["Open", "High", "Low", "Close"]

######################################################## Data Exploration and Analysis ########################################################
with tabs[0]:
    st.markdown("<font size=4><b>Data Exploration and Analysis: </b></font><font size=3>Get familiar with essentials about stock data.</font>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns([2,2,2,2,2,2 ])
    with col1:
        ticker = st.selectbox("Choose a Ticker", index=0, options = stock_tickers)

    with col2:
        interval = st.selectbox("Time Granularity", index=0, options = intervals)

    with col3:
        days = 365 if interval=='1d' else 5
        from_time = st.date_input("From", value=datetime.now() + timedelta(days = -days),
                        min_value = datetime.now() + timedelta(days=-days-1),
                        max_value = datetime.now() + timedelta(days = 1))

    with col4:
        to_time = st.date_input("To", value=datetime.now() + timedelta(days = -1),
                        min_value = datetime.now() + timedelta(days=-days),
                        max_value = datetime.now() + timedelta(days = -1))

    with col5:
        price = st.selectbox("Price Type", index=3, options = prices)

    with col6:
        col6.markdown("")
        col6.markdown("")
        btn_load = st.button("Show")

    if btn_load:

        #### chart 1: candle stick
        df_ticker = get_one_ticker_df(ticker,interval )
        if df_ticker.empty:
            st.warning(f"Data not available for ticker {ticker}")
        else:
            col1, col2 = st.columns([9,1])
            with col1:
                st.markdown(f"<font size=5 color=blue><b> Candlestick chart for {ticker} from {from_time} to {to_time}</font>", unsafe_allow_html=True)
            col1, col2 = st.columns([9,1])
            with col1:
                df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
                df_ticker = df_ticker[(df_ticker['Date'].dt.date>=from_time) &(df_ticker['Date'].dt.date<=to_time) ]
                fig = go.Figure(data=[go.Candlestick(x=df_ticker['Date'],
                                open=df_ticker['Open'],
                                high=df_ticker['High'],
                                low=df_ticker['Low'],
                                close=df_ticker['Close'])])
                st.plotly_chart(fig, use_container_width=True)

        #### chart 2: sma
        visualize_sma_one_ticker(price, ticker,  interval=interval)
        st.pyplot(plt)
        plt.clf()

        #### chart 2:wma (or ewm)
        visualize_ewm_one_ticker(price, ticker,  interval=interval)
        st.pyplot(plt)
        plt.clf()

        #### chart 3:rsi
        plot_RSI_streamlit(ticker=ticker, interval=interval, price_type=price )
        st.pyplot(plt)
        plt.clf()


    st.markdown("""---""")
    st.markdown("<font color=blue>Regenerate ticker list and reload historical data: </font>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,2,5])
    with col1:
        btn_ticker_list = st.button("regenerate ticker list")
    with col2:
        btn_historical = st.button("reload historical data")

    if btn_ticker_list:
        choose_and_save_my_list(refresh_list=True)
        st.success("Successfully generate a new list of tickers")

    if btn_historical:
        get_histories()
        st.success("Successfully load historical data (1d, 1m and 5m)")

######################################################## RSI ########################################################
with tabs[1]:
    st.markdown("<font size=4><b>Trade Strategies RSI </b></font><font size=3>Check and Inspect RSI and ARIMA Positioning.</font>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns([2,2,2,2,2,2 ])
    with col1:
        ticker_position = st.selectbox("Choose a Ticker for Positioning", index=0, options = stock_tickers)
    with col2:
        interval_position = st.selectbox("Time Positoining Granularity", index=0, options = intervals)
    with col3:
        price_rsi = st.selectbox("Price Type RSI", index=3, options = prices)
    with col4:
        col4.markdown("")
        col4.markdown("")
        btn_load_position = st.button("Inspect and Check")

    #### markdown 1: checking
    st.markdown("<font color=blue size=5><b>A. Inspect Positioning Using RSI</b></font>", unsafe_allow_html=True)

    if btn_load_position:
        # trading points
        plot_trading_points(ticker=ticker_position,interval=interval_position, price_type=price_rsi)
        st.pyplot(plt)
        plt.clf()

        # positioning
        create_plot_position(ticker=ticker_position,interval=interval_position, price_type=price_rsi)
        st.pyplot(plt)
        plt.clf()

######################################################## ARMIA ########################################################
with tabs[2]:
    st.markdown("<font size=4><b>Trade Strategies ARIMA </b></font><font size=3>Check and Inspect ARIMA Positioning.</font>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns([2,2,2,2,2,2 ])
    with col1:
        ticker_arima = st.selectbox("Choose a Ticker for ARIMA Positioning", index=0, options = stock_tickers)
    with col2:
        interval_arima = st.selectbox("Time Positoining ARIMA Granularity", index=0, options = intervals)
    with col3:
        price_arima = st.selectbox("Price Type Arima", index=3, options = prices)
    with col4:
        col4.markdown("")
        col4.markdown("")
        btn_load_arima = st.button("Inspect and Check ARIMA")

    #### markdown 1: checking
    st.markdown("<font color=blue size=5><b>A. Inspect Positioning Using ARIMA</b></font>", unsafe_allow_html=True)

    if btn_load_arima:
        # stock price
        plot_stock(ticker=ticker_arima, interval=interval_arima)
        st.pyplot(plt)
        plt.clf()

        # trading plot stock
        dftest, dfoutput  = test_stationarity(ticker=ticker_arima,interval=interval_arima, price_type="Close")
        st.pyplot(plt)
        plt.clf()
        # additional inf
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        dfoutput.columns = ['Values']
        # st.dataframe(dfoutput)
        p_value = dfoutput['p-value']
        if p_value < 0.05:
            st.markdown(f"#### result : p-value is {'{:.2f}'.format(p_value)}. This time series is stationary")
        else :
            p_value = {':.2f'}.format(p_value)
            st.markdown(f"#### result : p-value is {'{:.2f}'.format(p_value)}. This time series is not stationary")

        # check seasonality
        check_trend_seasonality(ticker=ticker, interval=interval, price_type=price_arima,
                                plot_percentage_change=False)
        st.pyplot(plt)
        plt.clf()

        # auto arima
        forecast, autoarima_summary, order = train_autoarima(ticker=ticker_arima,
                     interval=interval_arima,
                     price_type=price_arima,
                     plot_percentage_change=False)
        # st.pyplot(plt)
        # plt.clf()

        st.write(autoarima_summary)

        st.markdown(f"### Signature of the this time series: ")
        p,d,q = order
        st.markdown(f"{p}: Lags in autoregressive component.")
        st.markdown(f"{d}: Number of times differenced.")
        st.markdown(f"{q}: Lags in moving average.")

        st.markdown(f"### Forecast for next 5 days: ")
        st.write(forecast.values)