import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import sqlite3
import time

st.set_page_config(page_title="Autonomous Curve Agent", layout="wide")
st.title("📈 Autonomous Curve Agent")

DB_FILE = "predictor_database.db"

# Cached data fetcher
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    return yf.Ticker(ticker).history(start="2020-01-01", end="2026-06-19")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("CREATE TABLE IF NOT EXISTS history (ticker TEXT, predicted REAL, actual_price REAL, target_date TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS forecasts (ticker TEXT, day_offset INTEGER, predicted REAL, actual_price REAL, target_date TEXT)")
    conn.commit()
    conn.close()

def update_past_actuals():
    conn = sqlite3.connect(DB_FILE)
    for table in ['history', 'forecasts']:
        try:
            cursor = conn.execute(f"SELECT rowid, ticker, target_date FROM {table} WHERE actual_price = 0.0 AND target_date < ?", (str(pd.Timestamp.now().date()),))
            for row in cursor.fetchall():
                rowid, ticker, target_date = row
                hist = yf.Ticker(ticker).history(start=target_date, end=(pd.to_datetime(target_date) + pd.Timedelta(days=1)))
                if not hist.empty:
                    actual = hist['Close'].iloc[0]
                    conn.execute(f"UPDATE {table} SET actual_price = ? WHERE rowid = ?", (actual, rowid))
                    time.sleep(1.5) # Pace the requests to avoid being blocked
        except: continue
    conn.commit()
    conn.close()

init_db()

ticker = st.text_input("Enter Ticker Symbol (e.g., MSFT):").upper()

if st.button("Run Autonomous Cycle"):
    if not ticker:
        st.error("Please enter a ticker.")
    else:
        update_past_actuals()
        try:
            data = get_stock_data(ticker)
            current_price = data['Close'].iloc[-1]
            
            # Adaptive Model Logic
            df = data[['Close']].copy()
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df = df.dropna()
            
            hist_vol = df['Log_Ret'].std()
            ideal_lookback = int(np.clip(100 / (hist_vol * 10), 50, 300))
            
            df_sync = df.tail(ideal_lookback).copy()
            y = df_sync['Log_Ret'].values
            X = np.arange(len(y)).reshape(-1, 1)
            
            model = SVR(kernel='rbf', C=10, gamma='scale').fit(X, y)
            
            # Forecast Logic
            forecast_prices = []
            last_p = current_price
            conn = sqlite3.connect(DB_FILE)
            for i in range(1, 6):
                pred_return = model.predict(np.array([[len(y) + i - 1]]))[0]
                last_p = last_p * np.exp(pred_return)
                forecast_prices.append(last_p)
                target_date = str((pd.Timestamp.now() + pd.Timedelta(days=i)).date())
                conn.execute("INSERT INTO forecasts VALUES (?, ?, ?, ?, ?)", (ticker, i, float(last_p), 0.0, target_date))
            conn.commit()
            conn.close()
            
            st.success(f"Forecast complete for {ticker}!")
            st.line_chart(pd.DataFrame(forecast_prices, columns=["Predicted Price"]))
            for i, p in enumerate(forecast_prices):
                st.write(f"Day {i+1}: **${p:.2f}**")
        except Exception as e:
            st.error(f"Autonomous Error: {e}")
