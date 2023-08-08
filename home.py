import streamlit as st
import pandas as pd
import json
import re

st.set_page_config(page_title= "Camp 1: programming, AL and Stock Trade", page_icon = "🔢", layout="wide")

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

st.title("🔢 Stempro/Trovapages Programming, AI and Stock Trade Summer Camp")

st.markdown("#### Load alpaca api key and secret. Store them in session state so that other pages can use it as well.")

# st.session_state.update(st.session_state) # only need when run in cloud

if "API_KEY" in st.session_state:
    st.markdown("alpaca api key and secret have already been loaded")
    reload = st.button("re-load/refresh api key/secret")
    if reload:
        del st.session_state["API_KEY"]
        del st.session_state["API_SECRET"]
else:
    key_file = st.file_uploader("upload alpaca key/secret file", type={"json"})
    if key_file is not None:
        key_file_json = json.load(key_file)

        has_all_info = 0
        if "API_KEY" in key_file_json:
            API_KEY = key_file_json["API_KEY"]
            st.session_state.API_KEY = API_KEY
            has_all_info += 1
        if "API_SECRET" in key_file_json:
            API_SECRET = key_file_json["API_SECRET"]
            st.session_state.API_SECRET = API_SECRET
            has_all_info += 1
        if "END_POINT" in key_file_json:
            END_POINT = key_file_json["END_POINT"]
            st.session_state.END_POINT = END_POINT
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
