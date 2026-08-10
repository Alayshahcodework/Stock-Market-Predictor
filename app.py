from flask import Flask, jsonify, request
from flask_cors import CORS

from services.stock_service import StockAnalysisService
from services.model_service import LogisticRegressionPredictor
from services.news_service import NewsSentimentService

app = Flask(__name__)
CORS(app)

stock_service = StockAnalysisService()
model_service = LogisticRegressionPredictor()
news_service = NewsSentimentService()


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "stock-market-predictor"})


@app.get("/api/search")
def search_stocks():
    query = (request.args.get("query", "") or "").strip().lower()
    results = stock_service.search_symbols(query)
    return jsonify({"results": results})


@app.get("/api/analysis")
def analyze_stock():
    symbol = (request.args.get("symbol", "AAPL") or "AAPL").upper().strip()
    ma_window = int(request.args.get("ma_window", 20))

    analysis = stock_service.analyze_symbol(symbol=symbol, ma_window=ma_window)
    prediction = model_service.predict_signal(analysis["features"])

    sentiment = news_service.analyze(symbol=symbol, headlines=analysis.get("news", []))

    explanation = (
        f"{analysis['company']} is showing a {prediction['direction']} bias with "
        f"{prediction['confidence']:.0f}% confidence. The latest momentum is supported by "
        f"a {analysis['indicators']['rsi']:.1f} RSI, a MACD of {analysis['indicators']['macd']:.2f}, "
        f"and a volume change of {analysis['indicators']['volume_change']:.1f}%."
    )

    return jsonify(
        {
            "symbol": symbol,
            "company": analysis["company"],
            "price": analysis["price"],
            "previous_close": analysis["previous_close"],
            "moving_averages": analysis["moving_averages"],
            "indicators": analysis["indicators"],
            "sentiment": sentiment,
            "news": analysis.get("news", []),
            "quarter_snapshot": analysis.get("quarter_snapshot", {}),
            "prediction": prediction,
            "explanation": explanation,
            "live_status": analysis["live_status"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
