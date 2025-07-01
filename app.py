import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import statsmodels.api as sm
import streamlit as st
import datetime
import tensorflow.keras.backend as K

# Load stock data using yfinance
def load_data(ticker):
    today = datetime.date.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start="2015-01-01", end=today)
    if df.empty:
        st.error("No data found for this ticker. Please check the symbol and try again.")
        return pd.DataFrame()
    df = df[['Close']].copy()
    df.dropna(inplace=True)
    return df

# Plot stock data
def plot_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'])
    plt.title("Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

# Linear Regression model
def linear_regression(df):
    df = df.copy()
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    df['LR_Pred'] = model.predict(X).ravel()
    df = df[['Close', 'LR_Pred']]
    return model, df

# LSTM model
def lstm_model(df, future_days=252):  # Default: 1 year
    K.clear_session()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(25, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=2, batch_size=32, verbose=0)

    pred_input = scaled_data[-60:]
    predictions = []
    for _ in range(future_days):
        pred_input_reshaped = pred_input.reshape(1, 60, 1)
        prediction = model.predict(pred_input_reshaped, verbose=0)
        predictions.append(prediction[0, 0])
        pred_input = np.append(pred_input[1:], prediction).reshape(60, 1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return model, predicted_prices

# ARIMA model
def arima_model(df, steps=1):
    try:
        model = sm.tsa.ARIMA(df['Close'], order=(5, 1, 0))
        result = model.fit()
        pred = result.forecast(steps=steps)
        if not isinstance(pred, (pd.Series, np.ndarray)):
            pred = pd.Series(pred)
        return result, pred
    except Exception as e:
        st.error(f"ARIMA Error: {e}")
        return None, pd.Series()

# Streamlit App
def main():
    st.title("📈 Stock Market Price Predictor")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", value="AAPL")
    if st.button("Predict"):
        df = load_data(ticker)
        if df.empty:
            return

        st.subheader("🔍 Raw Stock Data")
        st.dataframe(df.tail())

        plot_data(df)
        st.image("plot.png", caption="Stock Price Over Time")

        _, df_lr = linear_regression(df)

        if isinstance(df_lr.columns, pd.MultiIndex):
            df_lr.columns = ['_'.join(map(str, col)).strip() for col in df_lr.columns]

        st.subheader("📊 Linear Regression Prediction")
        st.line_chart(df_lr)

        st.subheader("📉 LSTM Predictions")
        years = st.selectbox("Select Forecast Period (Years)", [1, 3, 5])
        with st.spinner(f"Predicting next {years} year(s)..."):
            _, lstm_preds = lstm_model(df, future_days=252*years)
            if len(lstm_preds) > 0:
                st.success(f"{years} Year (LSTM): ${lstm_preds[-1]:.2f}")
            else:
                st.warning("LSTM prediction failed. Try with more data or another stock.")

        st.subheader("📈 ARIMA Predictions")
        with st.spinner(f"Forecasting with ARIMA for {years} year(s)..."):
            _, arima_preds = arima_model(df, steps=252*years)
            if isinstance(arima_preds, pd.Series) and not arima_preds.empty:
                st.success(f"{years} Year (ARIMA): ${arima_preds.iloc[-1]:.2f}")
            elif isinstance(arima_preds, np.ndarray) and len(arima_preds) > 0:
                st.success(f"{years} Year (ARIMA): ${arima_preds[-1]:.2f}")
            else:
                st.warning("ARIMA prediction failed. Try with more data or another stock.")

        st.caption("Prediction based on historical data up to the latest available date.")

if __name__ == '__main__':
    main()
