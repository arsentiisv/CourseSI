# LLM Analysis of Charts in Investments

A comprehensive AI-driven system for stock market forecasting that combines deep learning, traditional machine learning, and natural language processing, accessible through a Telegram bot interface.

## Overview

This project democratizes stock market analysis by providing automated investment insights for both novice and experienced investors. The system integrates multiple AI technologies to deliver comprehensive market analysis through an intuitive conversational interface.

### Key Capabilities

- **Financial Data Processing**: Automated collection and preprocessing of market data
- **Price Forecasting**: Advanced hybrid LSTM + Random Forest modeling for future price predictions
- **Sentiment Analysis**: Real-time analysis of financial news using state-of-the-art NLP models
- **Natural Language Insights**: GPT-powered interpretation of forecasts into actionable recommendations
- **User-Friendly Interface**: Accessible through Telegram bot with support for multiple companies

## Features

### Advanced Forecasting Engine
- **Hybrid ML Pipeline**: Combines LSTM neural networks with Random Forest using volatility-aware dynamic weighting
- **Technical Analysis**: 30+ indicators including RSI, MACD, Bollinger Bands, VWAP, OBV, and MFI
- **Confidence Intervals**: Bootstrapped predictions with uncertainty quantification
- **Performance Metrics**: Comprehensive evaluation using MAE, RMSE, MAPE, and R²

### Real-Time Sentiment Analysis
- **News Aggregation**: Automated parsing from leading Russian financial sources:
  - Lenta.ru
  - RBC.ru
  - AiF.ru
- **NLP Processing**: Sentiment analysis powered by ruBERT model
- **Market Impact Assessment**: Integration of news sentiment with technical analysis

### Intelligent Recommendations
- **GPT-4o Integration**: Natural language summaries of complex market data
- **Actionable Insights**: Clear Buy/Hold/Sell recommendations
- **Context-Aware Analysis**: Combines technical indicators with market sentiment

### Telegram Bot Interface
- **Conversational UI**: Natural language interaction for market queries
- **Multi-Company Support**: Analysis across different stocks and sectors
- **Flexible Input**: Various query formats and company identification methods

### Core Modules

| Module | Description |
|--------|-------------|
| `TelBot.py` | Telegram bot logic and user interface management |
| `RefTrain.py` | Complete ML pipeline and forecasting engine |
| `SentimentPart.py` | News parsing and sentiment analysis functionality |
| `botgptsentim.py` | GPT-based forecast interpretation and summary generation |

### External Integrations

- **Tinkoff Invest API**: Real-time and historical financial data
- **OpenAI GPT-4o**: Natural language processing and summary generation
- **News Sources**: Automated content aggregation from major financial outlets

## Model Architecture

### Forecasting Pipeline
1. **Data Preprocessing**: Feature engineering and normalization
2. **Feature Selection**: Recursive Feature Elimination with Random Forest
3. **Model Training**: Adaptive ensemble on 10+ years of historical data
4. **Prediction**: Recursive forecasting with confidence intervals
5. **Validation**: Cross-validation and performance evaluation

### Performance Characteristics
- **Training Data**: 10+ years of comprehensive market data
- **Update Frequency**: Real-time data integration
- **Accuracy Metrics**: Validated against multiple statistical measures
- **Confidence Scoring**: Uncertainty quantification for all predictions

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Telegram Bot Token
- Tinkoff Invest API Token
- OpenAI API Key (GPT-4o access)

### Installation

```bash
git clone https://github.com/arsentiisv/CourseSI.git
cd CourseSI
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root
2. Add your API credentials:
```env
TELEGRAM_BOT_TOKEN=your_telegram_token
TINKOFF_API_TOKEN=your_tinkoff_token
OPENAI_API_KEY=your_openai_key
```

### Launch

```bash
python TelBot.py
```

## Example Output

### GPT-Generated Forecast Summary
```
The model indicates a moderate uptrend with 80% confidence over the next 5 trading days. 
Recent news sentiment shows increased investor optimism, with positive coverage of quarterly 
earnings and sector developments. Technical indicators support the bullish outlook with RSI 
at 58 and MACD showing positive momentum. Recommendation: BUY
```

## Technology Stack

### Machine Learning & Data Science
- **scikit-learn**: Traditional ML algorithms and utilities
- **Keras/TensorFlow**: Deep learning framework for LSTM networks
- **pandas & numpy**: Data manipulation and numerical computing
- **plotly**: Interactive visualization and charting

### Natural Language Processing
- **transformers**: Hugging Face library for transformer models
- **ruBERT**: Russian language BERT model for sentiment analysis
- **OpenAI GPT-4o**: Large language model for text generation

### Infrastructure & APIs
- **telebot**: Telegram Bot API wrapper
- **Tinkoff Invest API**: Financial data provider
- **asyncio**: Asynchronous programming support

## Roadmap

### Short-term Goals
- [ ] International news sentiment analysis expansion
- [ ] English language query support
- [ ] Additional foreign market coverage
- [ ] Performance optimization for real-time responses

### Long-term Vision
- [ ] Web-based mini-application deployment
- [ ] Mobile application development
- [ ] Advanced portfolio management features
- [ ] Multi-language support expansion


## Team

### Core Developers
- **Mikhail Ivanov** - Technical analysis, ML model development, GPT integration
- **Arsentii Sergienko** - Data parsing, sentiment analysis, bot development

### Academic Supervision
- **Valentina Kovaleva** - Project supervision and guidance
- **Tamara Voznesenskaya** - Research methodology and validation

