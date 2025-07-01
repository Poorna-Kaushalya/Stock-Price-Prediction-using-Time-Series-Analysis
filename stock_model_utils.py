import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = 'aapl_data.csv'
MODEL_PATH = 'linear_regression_model.pkl'

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        if 'Date' not in df.columns:
            raise ValueError("CSV file does not contain 'Date' column.")
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except (FileNotFoundError, ValueError, pd.errors.ParserError):
        print("Fetching fresh data from Yahoo Finance...")
        df = yf.download("AAPL", start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=True)
        df.columns = df.columns.get_level_values(0)  # Ensure flat column names
        return df


def save_data(data):
    data.to_csv(DATA_PATH)

def fetch_new_data(existing_data):
    last_date = existing_data.index[-1]
    today = datetime.today().date()

    if last_date.date() >= today - timedelta(days=1):
        print("Data already up to date.")
        return existing_data

    new_data = yf.download("AAPL", start=last_date + timedelta(days=1), end=today + timedelta(days=1), auto_adjust=True)
    new_data.columns = new_data.columns.get_level_values(0)
    updated_data = pd.concat([existing_data, new_data])
    updated_data = updated_data[~updated_data.index.duplicated()]
    return updated_data

def preprocess_and_train(data):
    data['Open_p'] = data['Open'].shift(1)
    data['Close_Lag1'] = data['Close'].shift(1)
    data['MA7'] = data['Close'].rolling(5).mean()
    data['MA14'] = data['Close'].rolling(10).mean()
    data.dropna(inplace=True)

    X = data[['Open_p', 'Close_Lag1', 'MA7', 'MA14']]
    y = data['Close']
    X_train, _, y_train, _ = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")
