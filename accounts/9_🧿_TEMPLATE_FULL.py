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
import asyncio

plt.rcParams.update({'font.size': 8})

st.set_page_config(page_title= "Camp 2: In Action", page_icon = "🔢", layout="wide")

PROJECT_PATH = os.getcwd()
DATA_PATH = f"{PROJECT_PATH}/data"
SCRIPTS_PATH = f"{PROJECT_PATH}/scripts_template"
sys.path.append(PROJECT_PATH)

from alpaca.trading.client import TradingClient
import alpaca_trade_api as tradeapi

from scripts_template.generate_ticker_list import choose_and_save_my_list, get_ticker_list
from scripts_template.get_histories import download_histories, get_one_ticker_df

from scripts_template.ticker_eda import visualize_sma_one_ticker, visualize_ewm_one_ticker
from scripts_template.rs_rsi import plot_RSI, plot_RSI_streamlit
from scripts_template.trade_rsi_strategy import plot_trading_points, create_plot_position

from scripts_template.auto_arima import (
        plot_stock,
        test_stationarity,
        check_trend_seasonality,
        show_train_test,
        train_autoarima,
        arima_forecast)

from scripts_template.model_training import load_performance, rsi_model, arima_model, get_candidates

from scripts_template.alpaca_actions_simple import get_account, get_positions, get_pending_orders, submit_alpaca_order_simple, get_rsi, get_arima
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
           "🧿Trade Strategies ARIMA",
           "📈Model Training",
           "🔢Trading Zone",
           "📇Advanced Trading",
           "📚Alpaca Keys", "        "]

whitespace = 9
## Fills and centers each tab label with em-spaces
tabs = st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

# Shared data among all tabs:
stock_tickers = get_ticker_list()
intervals = ['1d', '1m', '5m']
prices = ["Open", "High", "Low", "Close"]


@st.cache_data
def get_tradeclient_api():
    API_KEY = st.session_state["TEMPLATE_API_KEY"]
    SECRET_KEY = st.session_state["TEMPLATE_API_SECRET"]
    END_POINT = st.session_state["TEMPLATE_END_POINT"]

    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    api =  tradeapi.REST(API_KEY,SECRET_KEY, END_POINT)

    return trading_client, api

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
        download_histories()
        st.success("Successfully load historical data (1d, 1m and 5m)")

######################################################## RSI ########################################################
with tabs[1]:
    st.markdown("<font size=4><b>Trade Strategies RSI </b></font><font size=3>Check and Inspect RSI Positioning.</font>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([2,2,2,2,2 ])
    with col1:
        ticker_position = st.selectbox("Choose a Ticker for Positioning", index=0, options = stock_tickers)
    with col2:
        interval_position = st.selectbox("Time Positoining Granularity", index=0, options = intervals)
    with col3:
        price_rsi = st.selectbox("Choose Price Type", index=3, options = prices)
    with col4:
        col4.markdown("")
        col4.markdown("")
        btn_load_position = st.button("Inspect RSI Positioning")

    #### markdown 1: checking
    st.markdown("<font color=blue size=5><b>Inspect Positioning Using RSI</b></font>", unsafe_allow_html=True)

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
    st.markdown("<font size=4><b>Trade Strategies ARIMA </b></font><font size=3>Check and Inspect ARIMA Modeling.</font>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([2,2,2,2,2 ])
    with col1:
        ticker_arima = st.selectbox("Choose a Ticker for ARIMA Modeling", index=0, options = stock_tickers)
    with col2:
        interval_arima = st.selectbox("Choose ARIMA Granularity", index=0, options = intervals)
    with col3:
        price_arima = st.selectbox("Select Price Type", index=3, options = prices)
    with col4:
        col4.markdown("")
        col4.markdown("")
        btn_load_arima = st.button("Inspect ARIMA Modeling")

    #### markdown 1: checking
    st.markdown("<font color=blue size=5><b>Inspect ARIMA Modeling</b></font>", unsafe_allow_html=True)

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


######################################################## model training ########################################################
with tabs[3]:
    st.markdown("<font size=4><b>Positioning, Model Training And Ranking</b></font><font size=3>", unsafe_allow_html=True)
    st.markdown("<font size=3><b>RSI Ranking By Performance</b></font>", unsafe_allow_html=True)

    @st.cache_data
    def load_():
        return load_performance("RSI"), load_performance("ARIMA")

    df_rsi, df_arima = load_()

    col1, col2, col3 , col4, col5 = st.columns([2,2,2,2,14])
    with col1:
        col1.markdown("")
        st.markdown("RSI Last Trained(UTC):")
    with col2:
        col2.markdown("")
        if df_rsi.shape[0]>0:
            trained_on = df_rsi.iloc[0]["trained_on"]
        st.markdown(f"<font color=blue><b>{trained_on}</b></font>", unsafe_allow_html=True)

    with col4:
        # col4.markdown("")
        # col4.markdown("")
        btn_train_rsi = st.button("Retrain RSI")
    st.dataframe(df_rsi)

    st.markdown("<font size=3><b>ARIMA Ranking By Performance</b></font>", unsafe_allow_html=True)
    col1, col2, col3 , col4, col5 = st.columns([2,2,2,2,8])
    with col1:
        st.markdown("ARIMA Last Trained(UTC)")
    with col2:
        if df_arima.shape[0]>0:
            trained_on = df_arima.iloc[0]["trained_on"]
        st.markdown(f"<font color=blue><b>{trained_on}</b></font>", unsafe_allow_html=True)

    with col4:
        # col4.markdown("")
        # col4.markdown("")
        btn_train_arima = st.button("Retrain ARIMA")
    st.dataframe(df_arima)

    if btn_train_rsi:
        st.warning("Retrain RSI...")
        rsi_model()

    if btn_train_arima:
        st.warning("Retrain ARIMA Model. It will take a few minutes to hours depending on how many tickers you are training")
        arima_model()

######################################################## Trading Zone ########################################################
with tabs[4]:
    st.markdown("<font size=5><b>Positions and Trading Zone. </b></font><font size=3> <font size=3><b>Paper Money Only</b></font>", unsafe_allow_html=True)

    st.markdown(f"### Action")
    alpaca_action = st.radio( "Action", ('Buy', 'Sell'), index=1, horizontal =True)

    if "TEMPLATE_API_KEY" not in st.session_state:
        st.warning(f"### Alpaca Paper Trade API KEY not available. Please load keys to proceed")
    else:

        def format_func_rsi(option):
                return rsi_candidates[option]
        def format_func_arima(option):
                return arima_candidates[option]

        def transpose_ref_data(model_, ref_data):
            N = len(ref_data)
            if model_ == "RSI": #
                cols = [f"RSI_{N-i-1}" for i in range(N) ]
            else:
                cols = [f"Price_{1+i}" for i in range(N) ]
            ref_data = {c:['{:.2f}'.format(d)] for c, d in zip(cols, ref_data)}



            return pd.DataFrame(ref_data)

        trading_client, api = get_tradeclient_api()

        _, portfolio, cashbalance = get_account(trading_client=trading_client)

        col1, col2, col3 , col4, col5 = st.columns([4,4,2,2,8])
        with col1:
            st.markdown(f"Your Overall Portfolios: <font color=blue><b>{portfolio} </b></font>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"Available Funding for Buy: <font color=blue><b>{cashbalance} </b></font>", unsafe_allow_html=True)

        @st.cache_data
        def get_performance_ranking_list():
            rsi_candidates, arima_candidates = get_candidates()
            return rsi_candidates, arima_candidates

        rsi_candidates, arima_candidates = get_performance_ranking_list()

        col1, col2, col3 , col4, col5 = st.columns([2,2,4,4, 6])
        with col1:
            model_ = st.selectbox("Model Type", options=("RSI","ARIMA"), index=0)
        with col2:
            granularity_ =  st.selectbox( 'Model Granularity:', ('1 day', "1 hour", "15 min", '5 min'), index=0)
        with col3:
            ticker_ = ""
            if model_ == "RSI":
                ticker_ = st.selectbox("(RSI)Select a Ticker", options=list(rsi_candidates.keys()), format_func=format_func_rsi)
            else:
                ticker_ = st.selectbox("(ARIMA)Select a Ticker", \
                        options=list(arima_candidates.keys()), format_func=format_func_arima)
        with col4:
            order_type_ = st.selectbox("Buy Order Type", options=("Limit", "Market"), index=1)

        ref =   "Recent RSI Readings" if model_=="RSI" else "Predictions"
        btn_ref = st.button(f"Load {ref}")

        if btn_ref:
            if model_ == "RSI":
                ref_data_  = get_rsi(api, ticker_, interval, price_type='Close', span=14, min_periods = 3, recent_n = 5 )
            else:
                ref_data_ = get_arima(api, ticker_, interval, price_type='Close', plot_percentage_change=False,predict_n=5)
            df_ref_ = transpose_ref_data(model_, ref_data_)
            st.session_state["df_ref_"] = df_ref_

        if "df_ref_" in st.session_state:
            st.markdown(f"#### {ref}")
            df_ref_ = st.session_state["df_ref_"]
            st.write(df_ref_)

        st.markdown(f"#### Place Order: ")
        col1, col2, col3, col4  = st.columns([2,2,3,10])
        with col1:
            price_ = st.number_input(f"{alpaca_action} Price", min_value=0.0, value=0.00, step=0.01, max_value=10000.0)
        with col2:
            shares_ = st.number_input( "Shares", min_value=0, value=0, max_value=100000)
        with col3:
            st.markdown("")
            st.markdown("")
            bt_place_order = st.button("Place Simple Order")

        if bt_place_order:
            status, order_msg = submit_alpaca_order_simple(
                    trading_client = trading_client,
                    api = api,
                    model = model_,
                    action = alpaca_action,
                    interval = granularity_,
                    ticker = ticker_,
                    order_type = order_type_,
                    order_valid = "Limit",
                    amount = 0,
                    rsi = 0,
                    price = price_,
                    shares = shares_
                )
            if status == 0:
                st.success(order_msg)
            else:
                st.warning(order_msg)


    ######################################################## Advanced Trading ########################################################
    with tabs[5]:
        st.markdown("<font size=5><b>Positions and Trading Zone. </b></font><font size=3> <font size=3><b>Paper Money Only</b></font>", unsafe_allow_html=True)

        st.markdown(f"### Action")
        alpaca_action = st.radio( "Action", ('Buy', 'Sell', 'Check Portfolios', 'Cancel An Order'), index=2, horizontal =True)

        if "TEMPLATE_API_KEY" not in st.session_state:
            st.warning(f"### Alpaca Paper Trade API KEY not available. Please load keys to proceed")
        else:
            trading_client, api = get_tradeclient_api()

            _, portfolio, cashbalance = get_account(trading_client=trading_client)
            df_positions = get_positions(trading_client=trading_client)

            col1, col2, col3 , col4, col5 = st.columns([4,4,2,2,8])
            with col1:
                st.markdown(f"Your Overall Portfolios: <font color=blue><b>{portfolio} </b></font>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"Available Funding for Buy: <font color=blue><b>{cashbalance} </b></font>", unsafe_allow_html=True)


            @st.cache_data
            def get_performance_ranking_list():
                rsi_candidates, arima_candidates = get_candidates()
                return rsi_candidates, arima_candidates

            @st.cache_data
            def get_pending():
                buying, selling = get_pending_orders( )
                return buying, selling

            def format_func_rsi(option):
                    return rsi_candidates[option]
            def format_func_arima(option):
                    return arima_candidates[option]

            if alpaca_action == "Check Portfolios":
                st.markdown(f"### My Positions in Alpaca")
                st.dataframe(df_positions)

                buying, selling = get_pending()
                st.markdown(f"#### Buying Orders Alpaca")
                st.dataframe(buying)

                st.markdown(f"#### Selling Orders in Alpaca")
                st.dataframe(selling)
            elif alpaca_action == "Cancel An Order":
                buying, selling = get_pending()
                st.markdown(f"#### Buying Orders Alpaca")
                st.dataframe(buying)

                st.markdown(f"#### Selling Orders in Alpaca")
                st.dataframe(selling)

                col1, col2, col3 = st.columns([2, 2, 8])
                with col1:
                    order_cancel = st.text_input("Enter an Order to Cancel", value="")
                with col2:
                    st.markdown("")
                    st.markdown("")
                    btn_cancel_order = st.button("Submit an Order Cancellation Request")

            else: ## Buy a=ir Sell
                if "TEMPLATE_API_KEY" not in st.session_state:
                    st.warning(f"### Alpaca Paper Trade API KEY not available. Please load keys to proceed")
                else:
                    rsi_candidates, arima_candidates = get_performance_ranking_list()
                    ################################################## Buy ##########################################
                    if alpaca_action == "Buy":
                        col1, col2, col3 , col4 = st.columns([2,2,4,4])
                        with col1:
                            buy_model = st.selectbox("Model", options=("RSI","ARIMA"), index=0)
                        with col2:
                            buy_model_granular =  st.selectbox( 'Granularity:', ('1 day', "1 hour", "15 min", '5 min'), index=0)
                        with col3:
                            buy_ticker = ""
                            if buy_model == "RSI":
                                buy_ticker = st.selectbox("RSI - Select a Ticker", options=list(rsi_candidates.keys()), format_func=format_func_rsi)
                            else:
                                buy_ticker = st.selectbox("ARIMA - Select a Ticker", \
                                        options=list(arima_candidates.keys()), format_func=format_func_arima)

                        col1, col2, col3 , col4, col5, col6, col7 = st.columns([3,3,3,3,3,4,8])
                        with col1:
                            buy_type = st.selectbox("The Order Type", options=("Limit", "Market"), index=1)
                        with col2:
                            buy_enforce = st.selectbox("Buy Order Good Till", options=("Today", "Good To Cancel"), index=0)
                        with col3:
                            buy_amount = st.number_input("Allocate Amount", min_value=1, max_value=100000)
                        with col4:
                            buy_price, buy_rsi = 0, 0
                            if buy_model=="ARIMA" and buy_type!="Market":
                                buy_price = st.number_input("Target Price (or below):", min_value=1, max_value=100000)
                            elif buy_model=="RSI" and buy_type!="Market":
                                buy_rsi = st.number_input("When RSI Reaches:", min_value=1, max_value=100, value=30)
                            else:
                                st.text_input("Buy Trade Condition", "(Not Applicable)")
                        with col5:
                            btn_buy_ref = False
                            st.markdown("")
                            st.markdown("")
                            btn_buy_ref  = st.button("Buy Model Suggestions")
                        with col6:
                            btn_buy = False
                            st.markdown("")
                            st.markdown("")
                            if btn_buy_ref:
                                btn_buy  = st.button("Submit Alpaca Order -- Buy")

                        if btn_buy_ref:
                            st.markdown(f"#### Model suggestions")
                            if buy_model == "ARIMA":
                                st.write(f"Forecast for next 5 ({buy_model_granular}) prices ")
                                df_arima_forecast = arima_forecast(ticker=buy_ticker, interval=buy_model_granular,
                                API_KEY=st.session_state.TEMPLATE_API_KEY,
                                API_SECRET=st.session_state.TEMPLATE_API_SECRET)
                                st.dataframe(df_arima_forecast)
                            else:
                                pass
                            #     plot_trading_points(ticker=rsi_buy_ticker,interval=buy_model_granular)
                            #     st.pyplot(plt)
                            #     plt.clf()
                            btn_buy = True
                        if btn_buy and 1==11:
                            status, message = submit_alpaca_order(
                                model = buy_model,
                                action = 'sell',
                                interval = buy_model_granular,
                                ticker = buy_ticker,
                                order_type = buy_type,
                                order_valid = buy_enforce,
                                amount = buy_amount,
                                rsi = buy_rsi,
                                price = buy_price,
                                shares = None
                            )
                            if status == 0:
                                st.success(message)
                            else:
                                st.warning(message)
                    ######################################################### Sell ###################################
                    else:
                        #if alpaca_action == "Buy":
                        col1, col2, col3 , col4 = st.columns([2,2,4,4])
                        with col1:
                            sell_model = st.selectbox("Choose Model", options=("RSI","ARIMA"), index=0)
                        with col2:
                            sell_model_granular =  st.selectbox( 'Select Granularity:', ('1 day', "1 hour", "15 min", '5 min'), index=0)
                        with col3:
                            sell_ticker = ""
                            if sell_model == "RSI":
                                sell_ticker = st.selectbox("Select a Ticker(RSI)", options=list(rsi_candidates.keys()), format_func=format_func_rsi)
                            else:
                                sell_ticker = st.selectbox("Select a Ticker(ARIMA)", \
                                        options=list(arima_candidates.keys()), format_func=format_func_arima)

                        col1, col2, col3 , col4, col5, col6 , col7 = st.columns([3,3,3,3,3,4,8])
                        with col1:
                            sell_type = st.selectbox("Sell Order Type", options=("Limit",  "Market"), index=1)
                        with col2:
                            sell_enforce = st.selectbox("Sell Order Good Till", options=("Today", "Good To Cancel"), index=0)
                        with col3:
                            sell_shares = st.number_input("Shares to Sell:", min_value=1, max_value=100000)
                        with col4:
                            sell_price, buy_rsi = 0, 0
                            if sell_model=="ARIMA" and sell_type!="Market":
                                buy_price = st.number_input("Target Price (or above):", min_value=1, max_value=100000)
                            elif sell_model=="RSI" and sell_type!="Market":
                                sell_rsi = st.number_input("When RSI Passes:", min_value=1, max_value=100, value=30)
                            else:
                                st.text_input("Sell Trade Condition", "(Not Applicable)")
                        with col5:
                            btn_sell_ref = False
                            st.markdown("")
                            st.markdown("")
                            btn_sell_ref  = st.button("Model Suggestions")
                        with col6:
                            btn_sell = False
                            st.markdown("")
                            st.markdown("")
                            if btn_sell_ref:
                                btn_sell  = st.button("Submit Alpaca Order -- Sell")
                        if btn_sell_ref:
                            st.markdown(f"#### Model suggestions")
                            if sell_model == "ARIMA":
                                df_arima_forecast = arima_forecast(ticker=sell_ticker,
                                interval=sell_model_granular,
                                API_KEY=st.session_state.TEMPLATE_API_KEY,
                                API_SECRET=st.session_state.TEMPLATE_API_SECRET )
                            else:
                                pass
                                # plot_trading_points(ticker=rsi_sell_ticker,interval=sell_model_granular, price_type="Close")
                                # st.pyplot(plt)
                                # plt.clf()
                            btn_sell = True

                        if btn_sell and 1==123:
                            status, message = submit_alpaca_order(
                                model = sell_model,
                                action = 'buy',
                                interval = sell_model_granular,
                                ticker = sell_ticker,
                                order_type = sell_type,
                                order_valid = sell_enforce,
                                amount = None,
                                rsi = sell_rsi,
                                price = sell_price,
                                shares = sell_shares
                            )
                            if status == 0:
                                st.success(message)
                            else:
                                st.warning(message)



######################################################## Alpaca Keys  ########################################################
with tabs[6]:
    st.markdown("#### Load alpaca api key and secret.")

    # st.session_state.update(st.session_state) # only need when run in cloud

    if "TEMPLATE_API_KEY" in st.session_state:
        st.markdown("alpaca api key and secret have already been loaded")
        reload = st.button("re-load/refresh api key/secret")
        if reload:
            del st.session_state["TEMPLATE_API_KEY"]
            del st.session_state["TEMPLATE_API_SECRET"]
            del st.session_state["TEMPLATE_END_POINT"]
    else:
        col1, col2 = st.columns([3, 5])
        with col1:
            key_file = st.file_uploader("upload alpaca key/secret file", type={"json"})
        if key_file is not None:
            key_file_json = json.load(key_file)

            has_all_info = 0
            if "TEMPLATE_API_KEY" in key_file_json:
                API_KEY = key_file_json["TEMPLATE_API_KEY"]
                st.session_state.TEMPLATE_API_KEY = API_KEY
                has_all_info += 1
            if "TEMPLATE_API_SECRET" in key_file_json:
                API_SECRET = key_file_json["TEMPLATE_API_SECRET"]
                st.session_state.TEMPLATE_API_SECRET = API_SECRET
                has_all_info += 1
            if "TEMPLATE_END_POINT" in key_file_json:
                END_POINT = key_file_json["TEMPLATE_END_POINT"]
                st.session_state.TEMPLATE_END_POINT = END_POINT
                has_all_info += 1

            if has_all_info == 3:
                st.markdown("### Successfully load alpaca key, secret and endpoint ")
                masked = re.sub('\w', '*', API_KEY[:-4])
                st.markdown(f"API_KEY --- {masked + API_KEY[-4:]}")
                masked = re.sub('\w', '*', API_SECRET[:-4])
                st.markdown(f"API_SECRET --- {masked + API_SECRET[-4:]}")
                st.markdown(f"END_POINT --- {END_POINT}")
            else:
                st.warning('Wrong alpaca secret file or format incorrect', icon="⚠️")

    # ##### async running to execute RSI order submit
    # async def execution_orders():
    #     await asyncio.sleep(300)  # execute once every 5 minutes
    #     alpaca_order_execution()
