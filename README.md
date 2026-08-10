# MarketPulse Predictor

MarketPulse Predictor is a full-stack stock market analysis app that combines a Flask backend with a Vite + React frontend. It provides a polished dashboard for exploring stocks, reviewing moving averages, analyzing RSI and MACD, viewing volume changes, and receiving a simple prediction outlook based on a logistic-regression-inspired scoring approach.

## Features

- Stock search with live-style suggestions
- Technical analysis with:
  - 5-day moving average
  - 20-day moving average
  - 1-month moving average
  - 1-year moving average
- Momentum and volatility indicators:
  - RSI
  - MACD
  - Volume change
  - Recent price momentum
- Sentiment-themed news summary
- Prediction explanation panel
- Modern, sleek UI with loading states and animated workflow feedback

## Project Structure

```text
backend/
  app.py
  requirements.txt
  services/
    model_service.py
    news_service.py
    stock_service.py
frontend/
  index.html
  package.json
  vite.config.js
  src/
    App.jsx
    main.jsx
    styles.css
```

## Tech Stack

### Backend
- Python
- Flask
- Flask-CORS
- pandas
- numpy
- scikit-learn
- yfinance

### Frontend
- React
- Vite
- CSS

## Installation

### 1. Clone the project

```bash
git clone <your-repository-url>
cd StockMarket2
```

### 2. Backend setup

```bash
cd backend
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend setup

```bash
cd ../frontend
npm install
```

## Running the App

### Start the backend

```bash
cd backend
py -3 app.py
```

The Flask API will run at:
- http://127.0.0.1:5000

### Start the frontend

```bash
cd frontend
npm run dev
```

The Vite app will run at:
- http://localhost:5173/

## API Endpoints

### Health check

```http
GET /api/health
```

### Stock search suggestions

```http
GET /api/search?query=AAPL
```

### Stock analysis

```http
GET /api/analysis?symbol=AAPL
```

## How the Prediction Works

The backend calculates a set of technical features from recent market data and passes them into a lightweight logistic-regression-based prediction layer. The model returns a simple bullish or bearish direction and a confidence score based on the feature values.

## Notes

- The app uses Yahoo Finance data when available and falls back to a built-in demo dataset if external data is unavailable.
- The project is intended as a functional prototype and can be expanded with richer modeling, charting, and historical training pipelines.

## Future Improvements

Possible enhancements include:
- Real historical training datasets
- Interactive price charts
- More advanced predictive models
- Better news sentiment integration
- User authentication and saved watchlists

## License

This project is provided as a learning and prototyping project.
