"""Функции принимают датасет и возвращают json данные"""
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# data = {
#     'UserSenderId': ['Twitter', 'Facebook', 'Twitter', 'Instagram', 'Facebook'],
#      'SubmitDate': ['2023-12-01 14:30:00', '2023-10-02 09:15:00', '2023-02-02 18:45:00', '2023-10-03 12:00:00', '2023-10-03 23:59:00'],
#     'MessageText': ["Отличный день!", "Все плохо...", "Нормально", "Я счастлив!", "Мне грустно"],
#      'Sentiment': ['G', 'B', 'N', 'G', 'B']
#  }
# df = pd.DataFrame(data)


def bar_chart(dataset: pd.DataFrame):
    """Данные для столбчатой диаграммы"""
    labels = ['B', 'N', 'G']
    sentiment_counts = dataset['Sentiment'].value_counts()  # Подсчет количества каждого элемента
    sentiment_counts = sentiment_counts.reindex(labels, fill_value=0)  # Убедимся, что все метки присутствуют
    counts_list = sentiment_counts[labels].tolist()  # Преобразуем в список

    # Рассчитываем проценты
    total = sum(counts_list)
    percentages = [round((count / total) * 100, 2) if total > 0 else 0 for count in counts_list]

    # Возврат данных для столбчатой диаграммы
    return {
        "labels": labels,  # Метки для графика
        "counts": counts_list,  # Количество
        "percentages": percentages  # Проценты
    }


def text_lengths_by_tone(dataset: pd.DataFrame):
    """Статистика длины текстов по тональности"""
    dataset['TextLength'] = dataset['MessageText'].apply(len) # Столбец с длиной текста
    length_stats = {}
    for label in ['B', 'N', 'G']:
        length_stats[label] = {
            'mean': dataset[dataset['Sentiment'] == label]['TextLength'].mean(),
            'median': dataset[dataset['Sentiment'] == label]['TextLength'].median(),
            'min': dataset[dataset['Sentiment'] == label]['TextLength'].min(),
            'max': dataset[dataset['Sentiment'] == label]['TextLength'].max()
    }
    del dataset['TextLength'] # Удаление
    return length_stats


def top_words_by_tone(dataset: pd.DataFrame):
    """Функция для подсчета частых слов по тонам"""
    def get_top_words(texts, top_n=5):
        words = ' '.join(texts).split()
        words = [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word).lower() not in stopwords.words('russian')]
        return dict(Counter(words).most_common(top_n))

    # Топ-5 слов для каждой метки
    top_words_stats = {}
    for label in ['B', 'N', 'G']:
        top_words_stats[label] = get_top_words(dataset[dataset['Sentiment'] == label]['MessageText'])
    return top_words_stats


def tone_over_time(dataset: pd.DataFrame):
    """Тональность по временным меткам (округление по месяцам)"""
    # Преобразуем столбец 'Date' в datetime
    dataset['SubmitDate'] = pd.to_datetime(dataset['SubmitDate'])
    # Округляем даты до месяцев
    dataset['Month'] = dataset['SubmitDate'].dt.to_period('M')  # Или df['Date'].dt.floor('M')
    # Группировка по месяцу и тональности
    time_sentiment = dataset.groupby(['Month', 'Sentiment']).size().unstack(fill_value=0)
    # Преобразуем в словарь
    time_sentiment_dict = time_sentiment.to_dict(orient='index')
    del dataset['SubmitDate']
    return time_sentiment_dict


def users_tone(dataset: pd.DataFrame):
    """Тональности пользователей"""
    # Группировка по источнику и тональности
    source_sentiment = dataset.groupby(['UserSenderId', 'Sentiment']).size().unstack(fill_value=0)
    # Преобразуем в словарь
    source_sentiment_dict = source_sentiment.to_dict(orient='index')
    return source_sentiment_dict

# print(bar_chart(df))
# print(text_lengths_by_tone(df))
# print(top_words_by_tone(df))
# print(tone_over_time(df))
# print(users_tone(df))