# 📈 Stock Price Predictor using Streamlit

This is a Streamlit web application that predicts stock prices using three machine learning models:

- 🔹 Linear Regression  
- 🔹 LSTM (Long Short-Term Memory) Neural Network  
- 🔹 ARIMA (AutoRegressive Integrated Moving Average)

---

## 🚀 Features

- Fetches historical stock data using **Yahoo Finance**
- Plots closing price trend
- Predicts stock prices for **1 year, 3 years, and 5 years**
- Shows forecasted values from **Linear Regression, LSTM, and ARIMA**
- Easy-to-use **Streamlit web interface**

---

## 📦 Requirements

Install all required dependencies:

pip install -r requirements.txt
▶️ How to Run
streamlit run app.py
Open http://localhost:8501 in your browser.

📂 Project Structure
├── app.py                # Main Streamlit app
├── plot.png              # Saved matplotlib plot
├── requirements.txt      # Python dependencies
└── .gitignore            # Git ignore file
📊 Example Prediction
Enter a ticker like AAPL or TSLA
Select forecast period — 1 / 3 / 5 years
See predictions from all 3 models displayed interactively!

🧠 Models Used
Linear Regression: For simple trend lines

LSTM: Trained on past 60 days to predict future prices

ARIMA: Time-series forecasting using statsmodels

🛠️ Future Improvements
Add more models (e.g., Prophet, XGBoost)

Include volume and other technical indicators

Deploy on Streamlit Cloud

✅ requirements.txt
streamlit
yfinance
pandas
numpy
matplotlib
scikit-learn
keras
tensorflow
statsmodels
How to use:
Save this content into a file named requirements.txt in your project directory.

To install all dependencies, run:

pip install -r requirements.txt


🧑‍💻 Author
Dev Dhananjay Singh
📬 LinkedIn
🌐 GitHub

