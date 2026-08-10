from typing import Dict, List


class NewsSentimentService:
    def analyze(self, symbol: str, headlines: List[Dict[str, str]]) -> Dict[str, object]:
        positive_keywords = {"supportive", "constructive", "improving", "resilient", "favorable"}
        negative_keywords = {"risk", "downside", "pressure", "weak", "caution"}

        score = 0
        for item in headlines:
            text = (item.get("title", "") or "").lower()
            for keyword in positive_keywords:
                if keyword in text:
                    score += 1
            for keyword in negative_keywords:
                if keyword in text:
                    score -= 1

        if score > 0:
            tone = "Positive"
        elif score < 0:
            tone = "Negative"
        else:
            tone = "Neutral"

        return {
            "symbol": symbol,
            "tone": tone,
            "headline_count": len(headlines),
            "summary": f"Recent headlines for {symbol} are leaning {tone.lower()} with a balanced mix of sentiment and momentum cues.",
        }
