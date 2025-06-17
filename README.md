LLM Analysis of Charts in Investments
A comprehensive AI-driven system for stock market forecasting that combines deep learning, traditional machine learning, and natural language processing, accessible through a Telegram bot.

Overview
This project aims to make stock market analysis accessible for both novice and experienced investors by automating the following tasks:
* Collecting and preprocessing financial and news data
* Forecasting future stock prices using a hybrid LSTM + Random Forest model
* Performing sentiment analysis on recent financial news using ruBERT
* Summarizing forecasts into natural language recommendations with GPT-4o
* Presenting everything through an intuitive Telegram bot interface


Features
* Stock Forecasting: Hybrid ML pipeline using LSTM and Random Forest with volatility-aware dynamic weighting
* News Sentiment Analysis: Real-time parsing from top Russian sources (Lenta.ru, RBC.ru, AiF.ru) and analysis with ruBERT
* LLM Insights: GPT-4o generates human-readable summaries (Buy/Hold/Sell) based on forecast charts and sentiment scores
* Telegram Bot Interface: Accessible, user-friendly interaction with support for multiple companies and flexible input formats
* Technical Indicators: 30+ classic and volume-based features including RSI, MACD, Bollinger Bands, VWAP, OBV, MFI, etc.


System Architecture
Modules:
* TelBot.py – Telegram bot logic and user interface
* RefTrain.py – Full ML pipeline and forecasting engine
* SentimentPart.py – News parsers and sentiment analysis logic
* botgptsentim.py – GPT-based forecast interpretation and summary
* External API: Tinkoff Invest API for financial data
Forecasting Model Highlights:
* Adaptive ensemble (LSTM + RF) trained on 10+ years of market data
* Recursive Feature Elimination + Random Forest for optimal feature selection
* Confidence intervals through bootstrapped predictions
* Evaluation metrics: MAE, RMSE, MAPE, R²


Getting Started
Requirements
* Python 3.8+
* Telegram Bot Token
* Tinkoff Invest API Token
* OpenAI API Key (for GPT-4o access)


Installation
git clone https://github.com/arsentiisv/CourseSI.git
cd CourseSI
pip install -r requirements.txt


Run the Bot
python TelBot.py


GPT Forecast Example
"The model shows a moderate uptrend with 80% confidence. Positive sentiment from news headlines suggests increased investor optimism. Recommendation: Buy."

Technologies Used
* scikit-learn, keras, pandas, numpy
* transformers, plotly, telebot
* OpenAI GPT-4o, ruBERT, Tinkoff Invest API
* News parsing from: Lenta.ru, RBC.ru, AiF.ru


Future Work
* Add international news sentiment analysis
* Support English queries and foreign companies
* Deploy as a web mini-app
* Improve real-time responsiveness and LLM summaries


Authors
* Mikhail Ivanov – Technical analysis, ML model, GPT integration
* Arsentii Sergienko – Tinkoff data parsing, sentiment analysis, bot development

Supervised by:
* Valentina Kovaleva
* Tamara Voznesenskaya

