import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# Подготовка модели
model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # токенизация
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint) # загрузка модели


def analysis(text):
    """Анализирует текст и возвращает все вероятности"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
    return proba


def get_proba(data):
    """В зависимости от входных данных возвращает результат анализа текста"""
    if data is list[str]: # если текст велик
        results = [analysis(el) for el in data]
        n = len(results)
        proba = np.array([0, 0, 0])
        for score in results:
            proba += score
        proba /= np.array([n, n, n])
    else: proba = analysis(data)
    return proba


def get_sentiment(text: str, return_type: str = 'score-label', passing_threshold: float = 1/3, emoji: bool = False,
                  coefficient: float = 1.5, start_boost: float = 0.5) -> str | tuple[str, float] | np.ndarray:
    """Определяет тональность текста text. return_type может быть: 'label' - наибольшая метка, 'score' - числовая оценка,
    'score-label' - числовая оценка, переведенная в метку, 'proba' - вероятности для каждой метки. passing_threshold -
    порог для определения меток. emoji - поиск и учет эмотиконов"""
    # обработка текста
    data = preprocessing(text)
    proba = get_proba(data)
    # Учет эмотиконов
    if emoji:
        emoji_vector = find_emoticons(text, coefficient, start_boost)
        for el in range(3):
            if emoji_vector[el]:
                proba[el] += emoji_vector[el]
        if any(emoji_vector) != 0:
            proba /= [2, 2, 2]
    # Выбор типа возвращаемых данных
    lbl = ['B', 'N', 'G']
    if return_type == 'label':
        return lbl[proba.argmax()]
    if return_type == 'score':
        return proba.dot([-1, 0, 1])
    # Преимущественно
    if return_type == 'score-label':
        score = proba.dot([-1, 0, 1])
        if score > passing_threshold:
            return 'G', score
        if score > - passing_threshold:
            return 'N', score
        return 'B', score
    if return_type == 'proba-label':
        score = proba.dot([-1, 0, 1])
        if score > passing_threshold:
            return 'G', proba
        if score > - passing_threshold:
            return 'N', proba
        return 'B', proba
    return proba


def clean_html(text: str) -> str:
    """Очищает текст text от HTML-разметки"""
    if pd.isna(text):  # Проверка на NaN
        return text
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def split_sentence(text: str, window_size: int = 514) -> list[str]:
    """Разбивает большой текст на предложения"""
    sentences = []  # список предложений
    for sentence in re.split(r"(?<=[.!?…]) ", text):
        if len(sentence) > window_size: # если предложение больше окна
            short_sentences = []
            words = sentence.split()
            i = 0
            while i < len(words):
                short_sentence = ''
                while len(short_sentence + words[i]) < window_size:
                    short_sentence += words[i]
                    i+=1
                short_sentences.append(short_sentence)
            sentences.extend(short_sentences)
        else:
            sentences.append(sentence)
    return sentences


def preprocessing(text, window_size: int = 514) -> str | list[str]:
    """Предварительная обработка текстов"""
    text = clean_html(text)
    if len(text) > window_size:
        mini_text = split_sentence(text)
        return mini_text
    return text


def find_emoticons(text: str, coefficient: float = 1.5, start_boost: float = 0.5) -> list[float]:
    """Принимает текст text и ищет смайлики. coefficient - коэффициент увеличения значения тональности по направлению,
    в котором найден эмотикон. start_boost - стартовая прибавка к значению тональности при первом обнаружении. Возвращает вектор тональности"""
    vector = [0.0, 0.0, 0.0]
    # проверку на простые смайлики ')' и '('
    count_close = text.count(')')
    count_open = text.count('(')
    if count_close > count_open:
        vector[2] = start_boost
    elif count_open > count_close:
        vector[0] = start_boost

    positive_emoticons = [":)", ":-)", "=)", ":D", ":-D", "=D", ";)", ";-)", ";)", ":-)", ":>", ":->", ":]", ":-]",
                          "=]", ":}", ":-}", "=}", "(:", "(-:", "(=", "C:", "c:", "^_^", "^^", "^-^", "('v')", "(^_^)",
                          "(^^)", "(^-^)", "(*^_^*)", "(*^^*)", "(*^-^*)"]
    negative_emoticons = [":(", ":-(", "=(", ":'(", ":-'(", "='(", ">:(", ">:-(", ">=(", "D:", "D-:", "D=", ">:O",
                          ">:-O", ">=O", ">:|", ">:-|", ">=|", ":/", ":-/", "=/", ":\\", ":-\\", "=\\", ":S", ":-S",
                          "=S", ">:S", ">:-S", ">=S", ">:\\", ">:-\\", ">=\\", ">:[", ">:-[", ">=[", ">:{", ">:-{",
                          ">={", ">:'(", ">:-'(", ">='(", "T_T", "T.T", ":'-(", ":'-|", ":'-{", ":'-[", ":'-\\", ":'-/",
                          ":'-S", ":'-O", ":'-D"]
    # проверка на смайлики
    for word in text.split():
        if word in positive_emoticons:
            if not vector[2]:
                vector[2] = start_boost
            else:
                vector[2] *= coefficient
        elif word in negative_emoticons:
            if not vector[0]:
                vector[0] = start_boost
            else:
                vector[0] *= coefficient
    return vector


'''Нужно добавить лемманизацию имен!'''
# # Тесты
# start_time = time.time()
#
#
# # dataset
# dataset = pd.read_excel('dataset_comments_35.xlsx')
# sentiments = [get_sentiment(text, return_type='proba-label', emoji=False) for text in dataset['MessageText']] # Результат
# sentiments2 = [get_sentiment(text, return_type='proba-label', emoji=True) for text in dataset['MessageText']]
#
# print(time.time()-start_time)
# print(sentiments)
# dataset['Result'] = sentiments
# dataset['Result_Emoji'] = sentiments2
# count_True = 0
# for i in range(168):
#     if dataset['Class'][i] == dataset['Result'][i][0][0].upper():
#         count_True += 1
#     else:
#         print(dataset.iloc[i, [2, 3, 4]], '\n')
# print('-'*150)
# print('Result', count_True/168)
# count_True2 = 0
# for i in range(168):
#     if dataset['Class'][i] == dataset['Result_Emoji'][i][0][0].upper():
#         count_True2 += 1
#     else:
#         print(dataset.iloc[i, [2, 3, 4]], '\n')
# print('Result_Emoji', count_True2/168)
#
# result_data = pd.DataFrame({
#     "Имя": [],
#     "Счет": []
# })
# result_data.loc[len(result_data)] = ['Result', count_True/168]
# result_data.loc[len(result_data)] = ['Result_Emoji', count_True2/168]
#
# # for start in range(70, 101, 1):
# #     start /= 100
# #     for coef in range(110, 210, 10):
# #         coef /= 100
# #         sentiments = [get_sentiment(text, return_type='proba-label', emoji=True, coefficient=coef, start_boost=start) for text in dataset['MessageText']]
# #         count_True = 0
# #         for i in range(168):
# #             if dataset['Class'][i] == sentiments[i][0][0].upper():
# #                 count_True += 1
# #         print(f'{start} {coef}: {count_True/168}')
# #         result_data.loc[len(result_data)] = [f'{start} {coef}', count_True/168]
# #
# # result_data.to_excel('Result_data.xlsx')
#
#
# for threshold in range(10, 90, 1):
#     start = 0.7
#     coef = 1.5
#     threshold /= 100
#     sentiments = [get_sentiment(text, return_type='proba-label', passing_threshold=threshold, emoji=True, coefficient=coef, start_boost=start) for text in dataset['MessageText']]
#     count_True = 0
#     for i in range(168):
#         if dataset['Class'][i] == sentiments[i][0][0].upper():
#             count_True += 1
#     print(f'{start} {coef} {threshold}: {count_True/168}')
#     result_data.loc[len(result_data)] = [f'{start} {coef} {threshold}', count_True/168]
#
# result_data.to_excel('Result_data_threshold.xlsx')
# dataset.to_excel('Model1.xlsx')