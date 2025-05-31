import os
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import telebot
from tinkoff.invest import Client, CandleInterval
# from realShit import SentiAn, Parser
from RefTrain import StockPredictorApp


# Токены
BOT_TOKEN     = 'BOT_Token'
TINKOFF_TOKEN = 'T_TOKEN'
Ml=StockPredictorApp()
bot, user_data = telebot.TeleBot(BOT_TOKEN), {}

# ---------- служебные функции -------------------------------------------------
def simS(str1, str2, zn):
    len_str1 = len(str1)
    len_str2 = len(str2)
    str1 = str1.lower()
    str2 = str2.lower()
    dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]
    for i in range(len_str1 + 1):
        dp[i][0] = i
    for j in range(len_str2 + 1):
        dp[0][j] = j
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    if dp[len_str1][len_str2] <= zn:
        return 1
    return 0

def naiFi(name):
    with Client(TINKOFF_TOKEN) as cl:
        vr = {'ticker': [], 'figi': [], 'name': [], 'type': []}
        for t in ['shares', 'bonds', 'etfs', 'currencies']:
            for item in getattr(cl.instruments, t)().instruments:
                if simS(name, item.name,  min(len(name), len(item.name))  // 1.75) or \
                   simS(name, item.ticker, min(len(name), len(item.ticker))// 1.75):
                    vr['ticker'].append(item.ticker)
                    vr['figi'  ].append(item.figi)
                    vr['type'  ].append(t)
                    vr['name'  ].append(item.name)
    return pd.DataFrame(vr)

def perd(p): return p.units + p.nano/1e9

def svechkiT(fishka):
    global kon
    global na
    kon = datetime.now() + timedelta(days=1)
    na = kon - timedelta(days=10 * 365)
    da = {'date': [],
          'Open': [],
          'High': [],
          'Low': [],
          'Close': [],
          'Volume': []
          }
    with Client(TINKOFF_TOKEN) as cl:
        for candle in cl.get_all_candles(
                figi=fishka,
                from_=na,
                to=kon,
                interval=CandleInterval.CANDLE_INTERVAL_DAY
        ):
            da['date'].append(str(candle.time + timedelta(hours=3))[:19])
            da['Open'].append(perd(candle.open))
            da['High'].append(perd(candle.high))
            da['Low'].append(perd(candle.low))
            da['Close'].append(perd(candle.close))
            da['Volume'].append(candle.volume)
    da = pd.DataFrame(da)
    return da


def send_candle_chart(chat_id: int, data: pd.DataFrame) -> None:
    s=Ml.run(data)
    display_df_formatted=s[0][-40:]
    mae=s[1]
    rmse=s[2]
    mape=s[3]
    r2=s[4]

    """Строит и отправляет график пользователю."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_df_formatted.index,
        y=display_df_formatted['close'],
        name="История",
        line=dict(color="blue")
    ))

    # Тестовые прогнозы (forecast, где is_forecast == True и lower_ci/upper_ci is NaN)
    test_forecast_df = display_df_formatted[
        (display_df_formatted['is_forecast'] == True) & (display_df_formatted['lower_ci'].isna())]
    fig.add_trace(go.Scatter(
        x=test_forecast_df.index,
        y=test_forecast_df['forecast'],
        name="Тестовый прогноз",
        line=dict(color="orange")
    ))

    # Будущие прогнозы (forecast, где lower_ci/upper_ci не NaN)
    future_forecast_df = display_df_formatted[display_df_formatted['lower_ci'].notna()]
    fig.add_trace(go.Scatter(
        x=future_forecast_df.index,
        y=future_forecast_df['forecast'],
        name="Будущий прогноз",
        line=dict(color="green", width=3)
    ))

    # Доверительный интервал для будущих прогнозов
    fig.add_trace(go.Scatter(
        x=future_forecast_df.index.tolist() + future_forecast_df.index.tolist()[::-1],
        y=future_forecast_df['upper_ci'].tolist() + future_forecast_df['lower_ci'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='80% доверительный интервал',
        showlegend=True
    ))
    fig.update_layout(
        title=f"R2 - {r2}, MAE - {mae}, MAPE - {mape}, RMSE - {rmse}",
        xaxis_title="Дата",
        yaxis_title="Цена закрытия (RUB)",
        template="plotly_white"
    )



    temppics_dir = os.path.join(os.getcwd(), 'temppics')
    os.makedirs(temppics_dir, exist_ok=True)
    image_path = os.path.join(temppics_dir, f'candlestick_chart_{chat_id}.png')
    fig.write_image(image_path, engine='kaleido')

    with open(image_path, 'rb') as photo:
        bot.send_photo(chat_id, photo)
    os.remove(image_path)

# ---------- handlers ----------------------------------------------------------
@bot.message_handler(commands=['start'])
def start(msg):
    user_data[msg.chat.id] = {}
    bot.send_message(msg.chat.id, "Name:")

@bot.message_handler(func=lambda m: True)
def get_instrument(msg):
    global txt
    chat_id, txt = msg.chat.id, msg.text
    try:
        vrFi = naiFi(txt)
        if vrFi.empty:
            bot.send_message(chat_id, "Не удалось найти инструмент. Попробуй снова.")
            return
        user_data[chat_id]['vrFi'] = vrFi
        listing = "\n".join([f"{i+1}. {r['name']} - {r['ticker']}" for i, r in vrFi.iterrows()])
        bot.send_message(chat_id, f"\n{listing}\nNumber:")
        bot.register_next_step_handler(msg, select_instrument)
    except Exception as e:
        bot.send_message(chat_id, f"Ошибка: {e}")

def select_instrument(msg):
    chat_id = msg.chat.id
    try:
        idx  = int(msg.text) - 1
        vrFi = user_data[chat_id]['vrFi']
        if idx not in range(len(vrFi)):
            bot.send_message(chat_id, "Неверный номер. Попробуй снова.")
            bot.register_next_step_handler(msg, select_instrument)
            return

        figi = vrFi.iloc[idx]['figi']
        data = svechkiT(figi)
        if data.empty:
            bot.send_message(chat_id, "Не удалось получить данные.")
            return

        send_candle_chart(chat_id, data)
        final_message(chat_id)  # <--- вызываем дополнительную функцию
        return data
    except Exception as e:
        bot.send_message(chat_id, f"Ошибка: {e}")
        bot.register_next_step_handler(msg, select_instrument)

def final_message(chat_id: int):
    # parse = Parser()
    # na1=kon-timedelta(days=40)
    # tabl = parse.SAn(txt, na1, kon)
    #
    # sentic = SentiAn()
    # tabl = sentic.fin(tabl)
    # text = tabl.to_string(index=False)
    bot.send_message(chat_id,"Всё нажми на /start")

# Запуск бота
bot.polling(none_stop=True)