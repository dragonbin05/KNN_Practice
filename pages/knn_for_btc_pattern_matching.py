import pandas as pd
import streamlit as st
import pyupbit
from datetime import datetime

st.write("# KNN For :red[BTC Pattern] Matching")
today = datetime.now()

st.write(f"Today: `{today.strftime("%Y-%m-%d %H:%M:%S")}`")
if st.button("Load BTC OHLCV Data"):
    try:
        data = pd.read_csv("data/btc_ohlcv.csv", index_col=0)
        data.index = pd.to_datetime(data.index)

        today_9 = today.replace(hour=9, minute=0, second=0, microsecond=0)
        time_diff = (today_9.date() - data.index[-1].date()).days
        if time_diff > 0 and today > today_9:
            data = data.append(pyupbit.get_ohlcv("KRW-BTC", interval="day", count=time_diff))
            data.to_csv("data/btc_ohlcv.csv")

    except FileNotFoundError:
        data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=3500)
        data.to_csv("data/btc_ohlcv.csv")

    st.write(data)

