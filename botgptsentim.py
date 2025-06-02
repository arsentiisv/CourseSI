import os
import base64
from openai import OpenAI
import pandas as pd

class GPTAnalyzer:
    def __init__(self, api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def encode_image_to_data_uri(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"Error with the image: {e}")
            return None

    def analyze(self, image_path: str, sentiment_df: pd.DataFrame, ticker: str, metrics: dict) -> str:
        try:
            # кодировка изображения
            data_uri = self.encode_image_to_data_uri(image_path)
            if not data_uri:
                return "Erorr: Can't decode the image"

            sentiment_text = "Result of the sentiment analysis:\n"
            if sentiment_df.empty:
                sentiment_text += "No data for sentiment.\n"
            else:
                sentiment_text += sentiment_df[-10:][['datetime', 'score']].to_string(index=False)
                sentiment_text += "\n\nA positive score indicates positive news, a negative score indicates negative news."

            #метрики для анализа
            mae = metrics.get('mae', 0.0)
            rmse = metrics.get('rmse', 0.0)
            mape = metrics.get('mape', 0.0)
            r2 = metrics.get('r2', 0.0)

            # промпт
            prompt = (
                f"Analyze the stock price forecast chart and news sentiment analysis results. "
                f"Provide a brief description of the chart for the required company in your own words (2-3 sentences), then: "
                f"Fill in the following template, inserting appropriate words based on the analysis of the chart and news. "
                f"For describing trends, use blue lines (historical prices), orange lines (test forecast), "
                f"and green lines (future forecast) with confidence intervals. Consider the news background from sentiment analysis. "
                f"Model metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}. "
                f"News background:\n{sentiment_text}\n\n"
                f"🔎 Analysis of forecast chart for {ticker} stock\n\n"
                f"1. 📈 Historical dynamics: Stock price in recent weeks has demonstrated "
                f"[nature of dynamics — growth / decline / volatility / stability], which is visible from "
                f"[brief description of blue and orange lines].\n\n"
                f"2. 📊 Model quality assessment: On the test dataset, the model showed the following metrics:\n"
                f"   - MAE: {mae:.2f}\n"
                f"   - RMSE: {rmse:.2f}\n"
                f"   - MAPE: {mape:.2f}%\n"
                f"   - R²: {r2:.4f}\n"
                f"   These values indicate [assessment — high / medium / low] forecast accuracy.\n\n"
                f"3. 🔮 Future forecast: According to the model, [forecast nature — growth / decline / sideways movement] is expected. "
                f"The forecast is accompanied by [confidence interval assessment — narrow / wide], which indicates "
                f"[high / low] model confidence.\n\n"
                f"📌 Recommendation: *[BUY / HOLD / SELL]* — based on the forecast, model metrics, and news background."
            )

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    }
                ],
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error with GPT API: {e}")
            return "Error with analysis by GPT"