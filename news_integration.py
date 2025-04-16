# # news_integration.py
#
# import pandas as pd
# from datetime import datetime
# from transformers import pipeline
#
# # Импортируем функции для парсинга новостей и объединения данных из других модулей
# from news_analysis import LentaNews, RbkNews, AiFNews
# from data_fusion import merge_news_data
#
#
# def aggregate_news_sentiment(query, start_date, end_date):
#     """
#     Получает новости из нескольких источников, выполняет сентимент-анализ и агрегирует
#     значения сентимента для каждой даты.
#
#     Аргументы:
#         query (str): Строка для поиска новостей.
#         start_date (str): Начальная дата (YYYY-MM-DD).
#         end_date (str): Конечная дата (YYYY-MM-DD).
#
#     Возвращает:
#         DataFrame с колонками 'date' и 'sentiment', где значение sentiment — усреднённое
#         значение сентимента по новостям для данной даты.
#     """
#     # Получаем новости с каждого источника
#     lenta_df = LentaNews(query, start_date, end_date)
#     rbk_df = RbkNews(query, start_date, end_date)
#     aif_df = AiFNews(query, start_date, end_date)
#
#     # Объединяем данные из всех источников
#     news_df = merge_news_data(lenta_df, rbk_df, aif_df)
#
#     # Инициализируем сентимент-анализ через Hugging Face pipeline
#     sentiment_analyzer = pipeline("sentiment-analysis")
#
#     def get_sentiment(text):
#         result = sentiment_analyzer(text)[0]
#         # Если результат положительный, возвращаем положительный score, иначе отрицательный
#         return result['score'] if result['label'].upper() == "POSITIVE" else -result['score']
#
#     # Выполняем сентимент-анализ для каждого заголовка
#     news_df['sentiment'] = news_df['title'].apply(get_sentiment)
#
#     # Агрегируем сентимент по датам — вычисляем среднее значение для каждой даты
#     agg_news = news_df.groupby('date')['sentiment'].mean().reset_index()
#     return agg_news
#
#
# def integrate_news_with_stock(stock_df, news_query, start_date, end_date):
#     """
#     Объединяет данные по котировкам с агрегированными новостными сентиментальными показателями.
#
#     Аргументы:
#         stock_df (DataFrame): Исторические данные котировок (индекс должен быть датой в формате YYYY-MM-DD).
#         news_query (str): Запрос для поиска новостей (например, тикер компании).
#         start_date (str): Начальная дата (YYYY-MM-DD).
#         end_date (str): Конечная дата (YYYY-MM-DD).
#
#     Возвращает:
#         Объединённый DataFrame, где к исходным данным по котировкам добавлен столбец 'sentiment'
#         с усреднёнными значениями сентимента за соответствующую дату.
#     """
#     # Получаем агрегированные новости
#     news_agg = aggregate_news_sentiment(news_query, start_date, end_date)
#
#     # Приводим индекс stock_df к строковому формату дат (YYYY-MM-DD)
#     stock_df.index = pd.to_datetime(stock_df.index).strftime("%Y-%m-%d")
#
#     # Объединяем по дате — левое объединение, чтобы сохранить все данные по котировкам
#     merged_df = stock_df.merge(news_agg, left_index=True, right_on='date', how='left')
#
#     # Если для каких-либо дат отсутствует сентимент, заполняем значение предыдущим (forward fill)
#     merged_df['sentiment'] = merged_df['sentiment'].fillna(method='ffill')
#
#     # Устанавливаем колонку 'date' в качестве индекса
#     merged_df.set_index('date', inplace=True)
#
#     return merged_df
#
#
# # Пример использования модуля при запуске напрямую
# if __name__ == "__main__":
#     from data_loader import load_data  # Импорт функции загрузки котировок
#
#     ticker = "AAPL"
#     start_date = "2019-01-01"
#     end_date = "2023-01-01"
#
#     # Загрузка данных котировок
#     stock_df = load_data(ticker, start_date, end_date)
#
#     # Интеграция новостного сентимента (news_query можно задать как тикер или другое ключевое слово)
#     news_query = ticker
#     merged_stock = integrate_news_with_stock(stock_df, news_query, start_date, end_date)
#
#     # Вывод первых строк объединённых данных
#     print(merged_stock.head())
