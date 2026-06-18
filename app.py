import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sqlalchemy import create_engine, text
import time

# Page configuration
st.set_page_config(page_title="Autonomous Curve Agent", layout="wide")
st.title("📈 Autonomous Curve Agent")

# Database connection using Streamlit Secrets
engine = create_engine(st.secrets["DB_URL"])

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    return yf.Ticker(ticker).history(start="2020-01-01", end="2026-06-20")

def update_past_actuals():
    with engine.connect() as conn:
        for table in ['history', 'forecasts']:
            # Select pending updates (where actual_price is 0.0)
            query = text(f"SELECT ctid, ticker, target_date FROM {table} WHERE actual_price = 0.0 AND target_date < :today")
            rows = conn.execute(query, {"today": str(pd.Timestamp.now().date())}).fetchall()
            
            for row in rows:
                ctid, ticker, target_date = row
                hist = yf.Ticker(ticker).history(start=target_date, end=(pd.to_datetime(target_date) + pd.Timedelta(days=1)))
                if not hist.empty:
                    actual = hist['Close'].iloc[0]
                    update_query = text(f"UPDATE {table} SET actual_price = :actual WHERE ctid = :ctid")
                    conn.execute(update_query, {"actual": float(actual), "ctid": ctid})
                    time.sleep(1.5) # Pace requests
        conn.commit()

ticker = st.text_input("Enter Ticker Symbol (e.g., MSFT):").upper()

if st.button("Run Autonomous Cycle"):
    if not ticker:
        st.error("Please enter a ticker.")
    else:
        with st.spinner('Auditing database and forecasting...'):
            update_past_actuals()
            try:
                data = get_stock_data(ticker)
                current_price = data['Close'].iloc[-1]
                
                # Model Logic
                df = data[['Close']].copy()
                df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
                df = df.dropna()
                
                hist_vol = df['Log_Ret'].std()
                ideal_lookback = int(np.clip(100 / (hist_vol * 10), 50, 300))
                
                df_sync = df.tail(ideal_lookback).copy()
                y = df_sync['Log_Ret'].values
                X = np.arange(len(y)).reshape(-1, 1)
                model = SVR(kernel='rbf', C=10, gamma='scale').fit(X, y)
                
                # Forecast
                forecast_prices = []
                last_p = current_price
                with engine.connect() as conn:
                    for i in range(1, 6):
                        pred_return = model.predict(np.array([[len(y) + i - 1]]))[0]
                        last_p = last_p * np.exp(pred_return)
                        forecast_prices.append(last_p)
                        target_date = str((pd.Timestamp.now() + pd.Timedelta(days=i)).date())
                        
                        conn.execute(text("INSERT INTO forecasts (ticker, day_offset, predicted, actual_price, target_date) VALUES (:t, :d, :p, 0.0, :date)"), 
                                     {"t": ticker, "d": i, "p": float(last_p), "date": target_date})
                    conn.commit()
                
                st.success(f"Forecast complete for {ticker}!")
                st.line_chart(pd.DataFrame(forecast_prices, columns=["Predicted Price"]))
                
                # Display stored data from Cloud DB
                st.subheader("Database Audit Log")
                with engine.connect() as conn:
                    table_query = text("SELECT ticker, predicted, target_date FROM forecasts ORDER BY target_date DESC LIMIT 10")
                    results = conn.execute(table_query).fetchall()
                    st.table(pd.DataFrame(results, columns=["Ticker", "Predicted", "Date"]))
                    
            except Exception as e:
                st.error(f"Autonomous Error: {e}")
