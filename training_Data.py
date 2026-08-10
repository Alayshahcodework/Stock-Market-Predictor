"""
Trains the LogisticRegressionPredictor model used in model_service.py.

Run this file directly to:
  1. Download historical price data for a list of stocks
  2. Build the same features analyze_symbol() produces (return_1, rsi, macd, etc.)
  3. Label each day: did the price go up the next day? (1 = yes, 0 = no)
  4. Split the data by time (not randomly) into train/test
  5. Train a LogisticRegression model
  6. Print accuracy so you know if it's actually better than a coin flip
  7. Save the trained model to trained_model.pkl

Usage:
    python train_model.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Feature order MUST match the order used in model_service.py's predict_signal().
# If these two lists ever drift apart, predictions will be wrong even though
# nothing crashes -- the model will just be reading the wrong number into the
# wrong slot.
FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "return_20",
    "ma_spread",
    "rsi",
    "macd",
    "volume_change",
    "recent_change",
]

# Train across several stocks instead of just one. A model trained on one
# stock tends to just memorize that stock's quirks. More symbols = a model
# that generalizes better, at the cost of a longer download/train time.
TRAINING_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "WMT",
]


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series) -> pd.Series:
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26


def build_features_for_symbol(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Downloads history for one symbol and builds a feature row for every day."""
    data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
    if data is None or data.empty:
        print(f"  Skipping {symbol}: no data returned")
        return pd.DataFrame()

    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)
    ma_20 = close.rolling(window=20).mean()

    df = pd.DataFrame(index=data.index)
    df["return_1"] = close.pct_change()
    df["return_5"] = close.pct_change(5)
    df["return_20"] = close.pct_change(20)
    df["ma_spread"] = ma_20 - close
    df["rsi"] = calculate_rsi(close)
    df["macd"] = calculate_macd(close)
    df["volume_change"] = volume.pct_change() * 100
    df["recent_change"] = close.pct_change() * 100

    # Label: did the price close higher the NEXT trading day?
    # shift(-1) pulls tomorrow's price back to today's row.
    df["target"] = (close.shift(-1) > close).astype(int)
    df["symbol"] = symbol

    # Drop rows with NaN -- these come from the rolling windows needing a
    # few days of history before they can produce a real number, plus the
    # very last row, which has no "tomorrow" to label.
    return df.dropna()


def build_training_set(symbols: list) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        frame = build_features_for_symbol(symbol)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No data downloaded for any symbol. Check your internet connection.")

    combined = pd.concat(frames)
    combined = combined.sort_index()
    return combined


def time_based_split(df: pd.DataFrame, train_fraction: float = 0.8):
    """
    Splits by date, not randomly. Random splitting would let the model train
    on rows that come AFTER some of its test rows in time, which leaks
    future information in and makes accuracy look better than it really is.
    """
    split_date = df.index.sort_values().unique()
    cutoff = split_date[int(len(split_date) * train_fraction)]

    train = df[df.index < cutoff]
    test = df[df.index >= cutoff]
    return train, test


def main():
    print("Building training dataset...\n")
    dataset = build_training_set(TRAINING_SYMBOLS)
    print(f"\nTotal rows: {len(dataset)}")

    train, test = time_based_split(dataset)
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")

    X_train, y_train = train[FEATURE_COLUMNS], train["target"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target"]

    print("\nTraining LogisticRegression...")
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.3f}")
    print("\nDetailed report:")
    print(classification_report(y_test, predictions, target_names=["down", "up"]))

    # A sanity check: what if the model just always predicted "up"?
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"Baseline (always guess the majority class): {baseline:.3f}")
    if accuracy <= baseline + 0.01:
        print(
            "\nWarning: the model is barely beating a model that always guesses "
            "the majority class. That means the features aren't giving it much "
            "real signal yet. This is common for a first pass, not a bug."
        )

    joblib.dump(model, "trained_model.pkl")
    print("\nSaved trained model to trained_model.pkl")


if __name__ == "__main__":
    main()