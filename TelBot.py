import os
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import telebot
from tinkoff.invest import Client, CandleInterval
from RefTrain import StockPredictorApp
from SentimentPart import Parser,SentiAn
import time
import threading

BOT_TOKEN     = '7580262489:AAHCy1YK_WT6iriTBEyTVD5P9uZzu-yP_kw'
TINKOFF_TOKEN = 't.wvc9ZeUlJjxD_HBUE0qGA95v2M_df56u36SGiR0KkexEgx7TtTA1CYn5mdvrwF0X8jE5oWbGb4rPLPU-VKJ6LA'

Ml = StockPredictorApp()
parse=Parser()
sentic=SentiAn()
bot = telebot.TeleBot(BOT_TOKEN)
user_data = {}  # { chat_id: { 'vrFi': DataFrame, 'tiki': str, 'figi': str, 'data_df': DataFrame, 'kon': datetime, 'na': datetime, 'search':str } }


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
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return 1 if dp[len_str1][len_str2] <= zn else 0


def naiFi(name):
    with Client(TINKOFF_TOKEN) as cl:
        vr = {'ticker': [], 'figi': [], 'name': [], 'type': []}
        for t in ['shares', 'bonds', 'etfs', 'currencies']:
            for item in getattr(cl.instruments, t)().instruments:
                if simS(name, item.name,  min(len(name), len(item.name)) // 1.75) or \
                   simS(name, item.ticker, min(len(name), len(item.ticker)) // 1.75):
                    vr['ticker'].append(item.ticker)
                    vr['figi'].append(item.figi)
                    vr['type'].append(t)
                    vr['name'].append(item.name)
    return pd.DataFrame(vr)


def perd(p):
    return p.units + p.nano / 1e9


def svechkiT(figi):
    kon = datetime.now() + timedelta(days=1)
    na = kon - timedelta(days=10 * 365)
    da = { 'date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [] }
    with Client(TINKOFF_TOKEN) as cl:
        for candle in cl.get_all_candles(
            figi=figi,
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
    df = pd.DataFrame(da)
    return df, kon, na


def animate_loading(bot, chat_id, message_id, stop_event):
    frames = ["⏳ Loading", "⏳ Loading.", "⏳ Loading..", "⏳ Loading..."]
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=frame)
        except Exception:
            pass
        i += 1
        #скорость обновления
        time.sleep(5)

def send_candle_chart(chat_id: int) -> None:
    # 1. Отправляем первое сообщение “Loading”
    loading_msg = bot.send_message(chat_id=chat_id, text="⏳ Loading")

    # 2. Запускаем поток анимации
    stop_event = threading.Event()
    anim_thread = threading.Thread(
        target=animate_loading,
        args=(bot, chat_id, loading_msg.message_id, stop_event)
    )
    anim_thread.start()

    try:
        data = user_data[chat_id]['data_df']
        tiki = user_data[chat_id]['tiki']
        s = Ml.run(data)

        display_df_formatted = s[0][-40:]
        mae, rmse, mape, r2 = s[1], s[2], s[3], s[4]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=display_df_formatted.index,
            y=display_df_formatted['close'],
            name="Historic prices",
            line=dict(color="blue", width=2)
        ))
        test_df = display_df_formatted[
            (display_df_formatted['is_forecast'] == True) & (display_df_formatted['lower_ci'].isna())
        ]
        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=test_df['forecast'],
            name="Test forecast",
            line=dict(color="orange", width=2)
        ))
        future_df = display_df_formatted[display_df_formatted['lower_ci'].notna()]
        fig.add_trace(go.Scatter(
            x=future_df.index.tolist() + future_df.index.tolist()[::-1],
            y=future_df['upper_ci'].tolist() + future_df['lower_ci'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% Confidence interval'
        ))
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df['forecast'],
            name="Future forecast",
            line=dict(color="green", width=3)
        ))

        metrics_text = f"MAE: {mae:.2f} │ RMSE: {rmse:.2f} │ MAPE: {mape:.2f}% │ R²: {r2:.4f}"
        fig.update_layout(
            title=f"Forecast of {tiki} with confidence intervals",
            xaxis_title="Date",
            yaxis_title="Close Price",
            template="plotly_white",
            annotations=[dict(
                text=metrics_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor='center', yanchor='top',
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            )],
            margin=dict(b=100),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        # Сохраняем и отправляем изображение
        temppics_dir = os.path.join(os.getcwd(), 'temppics')
        os.makedirs(temppics_dir, exist_ok=True)
        image_path = os.path.join(temppics_dir, f'candlestick_chart_{chat_id}.png')
        fig.write_image(image_path, engine='kaleido')
        #Чистим путь для подключения чата это надо перенести после чата
        with open(image_path, 'rb') as photo:
            bot.send_photo(chat_id, photo)
        os.remove(image_path)

    finally:
        # 4. Тормозим и итоговое
        stop_event.set()
        anim_thread.join()

        try:
            bot.edit_message_text(chat_id=chat_id, message_id=loading_msg.message_id, text="✅ Done!")
        except Exception:
            pass

# def send_candle_chart(chat_id: int) -> None:
#     data = user_data[chat_id]['data_df']
#     tiki = user_data[chat_id]['tiki']
#     s = Ml.run(data)
#     display_df_formatted = s[0][-40:]
#     mae, rmse, mape, r2 = s[1], s[2], s[3], s[4]
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=display_df_formatted.index,
#         y=display_df_formatted['close'],
#         name="Historic prices",
#         line=dict(color="blue", width=2)
#     ))
#     test_df = display_df_formatted[
#         (display_df_formatted['is_forecast'] == True) & (display_df_formatted['lower_ci'].isna())
#     ]
#     fig.add_trace(go.Scatter(
#         x=test_df.index,
#         y=test_df['forecast'],
#         name="Test forecast",
#         line=dict(color="orange", width=2)
#     ))
#     future_df = display_df_formatted[display_df_formatted['lower_ci'].notna()]
#     fig.add_trace(go.Scatter(
#         x=future_df.index,
#         y=future_df['forecast'],
#         name="Future forecast",
#         line=dict(color="green", width=3)
#     ))
#     fig.add_trace(go.Scatter(
#         x=future_df.index.tolist() + future_df.index.tolist()[::-1],
#         y=future_df['upper_ci'].tolist() + future_df['lower_ci'].tolist()[::-1],
#         fill='toself',
#         fillcolor='rgba(0,255,0,0.2)',
#         line=dict(color='rgba(255,255,255,0)'),
#         name='80% Confidence interval'
#     ))
#
#     metrics_text = f"MAE: {mae:.2f} │ RMSE: {rmse:.2f} │ MAPE: {mape:.2f}% │ R²: {r2:.4f}"
#     fig.update_layout(
#         title=f"Forecast of {tiki} with confidence intervals",
#         xaxis_title="Date",
#         yaxis_title="Close Price",
#         template="plotly_white",
#         annotations=[dict(
#             text=metrics_text,
#             xref="paper", yref="paper",
#             x=0.5, y=-0.15,
#             xanchor='center', yanchor='top',
#             showarrow=False,
#             font=dict(size=12, color="black"),
#             bgcolor="rgba(255,255,255,0.8)",
#             bordercolor="rgba(0,0,0,0.1)",
#             borderwidth=1
#         )],
#         margin=dict(b=100),
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
#     )
#
#     temppics_dir = os.path.join(os.getcwd(), 'temppics')
#     os.makedirs(temppics_dir, exist_ok=True)
#     image_path = os.path.join(temppics_dir, f'candlestick_chart_{chat_id}.png')
#     fig.write_image(image_path, engine='kaleido')
#
#     with open(image_path, 'rb') as photo:
#         bot.send_photo(chat_id, photo)
#     os.remove(image_path)


@bot.message_handler(commands=['start'])
def start(msg):
    chat_id = msg.chat.id
    user_data[chat_id] = {}
    bot.send_message(chat_id, "Enter the name of the company\nor it's ticker:")


@bot.message_handler(func=lambda m: True)
def get_instrument(msg):
    chat_id, txt = msg.chat.id, msg.text.strip()
    try:
        vrFi = naiFi(txt)
        if vrFi.empty:
            bot.send_message(chat_id, "Не удалось найти инструмент. Попробуйте снова.")
            return
        user_data[chat_id]['vrFi'] = vrFi
        user_data[chat_id]['search'] = txt

        listing = "\n".join([
            f"{i+1}. {row['name']} — {row['ticker']}"
            for i, row in vrFi.iterrows()
        ])
        bot.send_message(chat_id, f"Similar companies:\n{listing}\n\nWrite the number:")
        bot.register_next_step_handler(msg, select_instrument)
    except Exception as e:
        bot.send_message(chat_id, f"Ошибка при поиске: {e}")


def select_instrument(msg):
    chat_id = msg.chat.id
    try:
        text = msg.text.strip()
        if not text.isdigit():
            bot.send_message(chat_id, "Пожалуйста, введите число, соответствующее выбору.")
            bot.register_next_step_handler(msg, select_instrument)
            return

        idx = int(text) - 1
        vrFi = user_data[chat_id].get('vrFi')
        if vrFi is None or idx < 0 or idx >= len(vrFi):
            bot.send_message(chat_id, "Wrong number. Use another number.")
            bot.register_next_step_handler(msg, select_instrument)
            return

        chosen_row = vrFi.iloc[idx]
        figi = chosen_row['figi']
        tiki = chosen_row['ticker']

        user_data[chat_id]['figi'] = figi
        user_data[chat_id]['tiki'] = tiki

        data_df, kon, na = svechkiT(figi)
        if data_df.empty:
            bot.send_message(chat_id, "No historic data. Use another name")
            return

        user_data[chat_id]['data_df'] = data_df
        user_data[chat_id]['kon'] = kon
        user_data[chat_id]['na'] = na

        send_candle_chart(chat_id)
        final_message(chat_id)

    except Exception as e:
        bot.send_message(chat_id, f"Ошибка при выборе инструмента: {e}")
        bot.register_next_step_handler(msg, select_instrument)


def final_message(chat_id: int):
    bot.send_message(chat_id, "Graph done. Restart command: /start")
    na=user_data[chat_id]['na'].strftime('%Y-%m-%d')
    kon = user_data[chat_id]['kon'].strftime('%Y-%m-%d')
    txt=user_data[chat_id]['search']
    tabl = parse.SAn(txt, na, kon)
    tabl = sentic.fin(tabl)
    print(tabl)
    print(txt)


bot.polling(none_stop=True)
