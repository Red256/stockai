import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime, timedelta
import alpaca_trade_api as alpaca
import plotly.graph_objects as go

custom_css = """
  <style>
    table {
      border-collapse: collapse;
      width: 100%
    }
    th, td {
      border : 1px solid black;
      padding: 8px;
      text-align: left;
      white-space: pre-wrap;
      word-wrap: break-word
    }
  </style>
"""

st.set_page_config(page_title= "Camp 1: Explore", page_icon = "🔢", layout="wide")
st.title("🔢 My StockAI Project")


st.markdown("""
    <style>
    .block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-right: 0rem;
    padding-left: 2rem;
    }
    </style>
    """)

######################################## tabs ########################################
listTabs =["🧑‍🏭Data Exploration", "🧑‍🎓Load/Processing", "📈 Model Training", "🔢 My Finished Product", "        "]


whitespace = 9
st.markdown("#### 💡 Checkout our exciting programs!")
## Fills and centers each tab label with em-spaces
tabs = st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

with tabs[0]:
    st.markdown("##### Data Exploration and Analysis")
    style = """
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            padding-bottom: 0.5em;
        }
        .subtitle {
            font-size: 24px;
            font-weight: bold;
            padding-bottom: 0.5em;
        }
        ul {
            margin-top: 0.5em;
            margin-bottom: 0.5em;
        }
    </style>
    """
    stock_tickers = ['AAPL', 'BA', 'CAT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'MSFT', 'INTL']
    time_frames = ['1D', '1H', '1W', '1M']

    st.markdown("Get familiar with essential stock data.")

    col1, col2, col3, col4, col5, col6 = st.columns([2,2,2,2,2,2])
    with col1:
        col1.markdown("")
        col1.markdown("")
        ticker = st.selectbox("Choose a Ticker", index=0, options = stock_tickers)

    with col2:
        col2.markdown("")
        col2.markdown("")
        time_frame = st.selectbox("Time Granularity", index=0, options = time_frames)

    with col3:
        col3.markdown("")
        col3.markdown("")
        from_time = st.date_input("Load From", value=datetime.now() + timedelta(days = -30),
                        min_value = datetime.now() + timedelta(days=-100),
                        max_value = datetime.now() + timedelta(days = -2))

    with col4:
        col4.markdown("")
        col4.markdown("")
        to_time = st.date_input("Load To", value=datetime.now() + timedelta(days = -1),
                        min_value = datetime.now() + timedelta(days=-100),
                        max_value = datetime.now() + timedelta(days = -1))

    with col5:
        col5.markdown("")
        col5.markdown("")
        col5.markdown("")
        col5.markdown("")
        btn_load = st.button("Call Alpaca")

    @st.cache_data
    def load_secrets():
        secret_ready = 0
        if 'API_KEY' in st.session_state:
            API_KEY = st.session_state.API_KEY
            secret_ready += 1
        if 'API_SECRET' in st.session_state:
            API_SECRET = st.session_state.API_SECRET
            secret_ready += 1
        if 'END_POINT' in st.session_state:
            END_POINT = st.session_state.END_POINT
            secret_ready += 1

        if secret_ready == 3:
            return API_KEY, API_SECRET, END_POINT
        else:
            return None, None, None

    API_KEY, API_SECRET, END_POINT = load_secrets()
    if not API_KEY:
        st.warning('Alpaca Key and Secret Not loaded. Go to Camp1 Page to Load them', icon="⚠️")
    else:
        api = alpaca.REST(API_KEY, API_SECRET, 'https://paper-api.alpaca.markets')
        stock_data = api.get_bars(ticker, timeframe = time_frame, start  = from_time,
        end = to_time,  limit=100).df

        # df =  api.get_bars(symbol= "AAPL", start= "2022-03-23",end= "2022-03-23",timeframe= "1Min").df
        st.markdown(f"Candlestick chart for {ticker} from {from_time} to {to_time}")
        fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                        open=stock_data['open'],
                        high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'])])

        st.plotly_chart(fig, use_container_width=True)



# # st.session_state.update(st.session_state)

# stock_tickers = ['AAPL', 'BA', 'CAT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'MSFT', 'INTL']
# time_frames = ['1D', '1H', '1W', '1M']

# st.markdown("Get familiar with essential stock data.")

# col1, col2, col3, col4, col5, col6 = st.columns([2,2,2,2,2,2])
# with col1:
#     col1.markdown("")
#     col1.markdown("")
#     ticker = st.selectbox("Choose a Ticker", index=0, options = stock_tickers)

# with col2:
#     col2.markdown("")
#     col2.markdown("")
#     time_frame = st.selectbox("Time Granularity", index=0, options = time_frames)

# with col3:
#     col3.markdown("")
#     col3.markdown("")
#     from_time = st.date_input("Load From", value=datetime.now() + timedelta(days = -30),
#                     min_value = datetime.now() + timedelta(days=-100),
#                     max_value = datetime.now() + timedelta(days = -2))

# with col4:
#     col4.markdown("")
#     col4.markdown("")
#     to_time = st.date_input("Load To", value=datetime.now() + timedelta(days = -1),
#                     min_value = datetime.now() + timedelta(days=-100),
#                     max_value = datetime.now() + timedelta(days = -1))

# with col5:
#     col5.markdown("")
#     col5.markdown("")
#     col5.markdown("")
#     col5.markdown("")
#     btn_load = st.button("Call Alpaca")

# @st.cache_data
# def load_secrets():
#     secret_ready = 0
#     if 'API_KEY' in st.session_state:
#         API_KEY = st.session_state.API_KEY
#         secret_ready += 1
#     if 'API_SECRET' in st.session_state:
#         API_SECRET = st.session_state.API_SECRET
#         secret_ready += 1
#     if 'END_POINT' in st.session_state:
#         END_POINT = st.session_state.END_POINT
#         secret_ready += 1

#     if secret_ready == 3:
#         return API_KEY, API_SECRET, END_POINT
#     else:
#         return None, None, None

# API_KEY, API_SECRET, END_POINT = load_secrets()
# if not API_KEY:
#     st.warning('Alpaca Key and Secret Not loaded. Go to Camp1 Page to Load them', icon="⚠️")
# else:
#     api = alpaca.REST(API_KEY, API_SECRET, 'https://paper-api.alpaca.markets')
#     stock_data = api.get_bars(ticker, timeframe = time_frame, start  = from_time,
#     end = to_time,  limit=100).df

#     # df =  api.get_bars(symbol= "AAPL", start= "2022-03-23",end= "2022-03-23",timeframe= "1Min").df
#     st.markdown(f"Candlestick chart for {ticker} from {from_time} to {to_time}")
#     fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
#                     open=stock_data['open'],
#                     high=stock_data['high'],
#                     low=stock_data['low'],
#                     close=stock_data['close'])])

#     st.plotly_chart(fig, use_container_width=True)
