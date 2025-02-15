"""Оптимальные настройки модели: return_type='score-label', passing_threshold: float=1/3, emoji=True, start_boost=0.7, coefficient=1.5, del_name=True, name_thresh: float=0.75"""
import torch
import time
import re
from transformers import AutoTokenizer, pipeline
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
import pymorphy3

nltk.download('punkt_tab')

nltk.download('punkt_tab')

# Инициализация анализатора
morph = pymorphy3.MorphAnalyzer()

# Подготовка модели
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=tokenizer)


def upload_smiles(name: str) -> set[str]:
    result = set([])
    with open(name, 'r', encoding="utf-8") as f:
        smiles = f.read().splitlines()
        for smile in smiles:
            if smile:
                result.add(smile)
    return result


# Смайлы
positive_emoticons = upload_smiles('emoji_positive.txt')
negative_emoticons = upload_smiles('emoji_negative.txt')


def analysis(text: str) -> np.ndarray:
    """Анализирует текст и возвращает все вероятности"""
    with torch.no_grad():
        proba = sentiment_task(text, return_all_scores=True)
    return np.array([vec['score'] for vec in proba[0]])


def get_proba(data: str | list[str]) -> np.ndarray:
    """В зависимости от входных данных возвращает результат анализа текста"""
    if isinstance(data, list): # если текст велик
        results = [analysis(el) for el in data]
        n = len(results)
        proba = np.array([0.0, 0.0, 0.0])
        for score in results:
            proba += score
        proba /= np.array([n, n, n])
    else:
        proba = analysis(data)
    return proba


def get_sentiment(text: str, return_type: str = 'score-label', passing_threshold: float = 1/3, emoji: bool = False,
                  coefficient: float = 1.5, start_boost: float = 0.7, del_name: bool = False, name_thresh: float = 0.75)\
        -> str | tuple[str, float] | np.ndarray:
    """Определяет тональность текста text. return_type может быть: 'label' - наибольшая метка, 'score' - числовая оценка,
    'score-label' - числовая оценка, переведенная в метку, 'proba' - вероятности для каждой метки. passing_threshold -
    порог для определения меток. emoji - поиск и учет эмотиконов"""
    #print(f'Начата обработка текста {text[:30]}')
    # обработка текста
    data = preprocessing(text, del_name=del_name, name_thresh=name_thresh)

    print(data)
    proba = get_proba(data)
    # Учет эмотиконов
    if emoji:
        emoji_vector = find_emoticons(data, coefficient, start_boost)
        for el in range(3):
            if emoji_vector[el]:
                proba[el] += emoji_vector[el]
        if any(emoji_vector) != 0:
            proba /= [2, 2, 2]
    # print(f'DATA: {data}\n{proba.dot([-1, 0, 1])}   {proba}')
    # Выбор типа возвращаемых данных
    lbl = ['B', 'N', 'G']
    if return_type == 'proba':
        return proba
    if return_type == 'label':
        return lbl[proba.argmax()]
    if return_type == 'score':
        return proba.dot([-1, 0, 1])
    # Преимущественно
    if return_type == 'score-label':
        score = proba.dot([-1, 0, 1])
        if score > passing_threshold:
            return 'G'
        if score > - passing_threshold:
            return 'N'
        return 'B'
    if return_type == 'proba-label':
        score = proba.dot([-1, 0, 1])
        if score > passing_threshold:
            return 'G', proba
        if score > - passing_threshold:
            return 'N', proba
        return 'B', proba
    return proba


def clean_html(text: str) -> str | None:
    """Очищает текст text от HTML-разметки"""
    if text is None or pd.isna(text):  # Проверка на NaN
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
            count = 0
            while count < len(words):
                short_sentence = ''
                while len(short_sentence + ' ' + words[count]) < window_size:
                    short_sentence += ' ' + words[count] if short_sentence else words[count]
                    count += 1
                short_sentences.append(short_sentence)
            sentences.extend(short_sentences)
        else:
            sentences.append(sentence)
    return sentences


def delete_name(text: str, prob_thresh: float = 0.75) -> str:
    """Удаляет имена"""
    for word in nltk.word_tokenize(text):
        for p in morph.parse(word):
            if 'Name' in p.tag and p.score >= prob_thresh:
                text = text.replace(word, '')
    # Удаление лишних пробелов
    text = ' '.join(text.split())
    return text


def preprocessing(text: str, window_size: int = 514, del_name: bool = False, name_thresh: float = 0.75) -> str | list[str]:
    """Предварительная обработка текстов"""
    text = clean_html(text)
    text = remove_path(text)
    text = remove_urls(text)
    if del_name:
        text = delete_name(text, prob_thresh=name_thresh)
    if len(text) > window_size:
        mini_text = split_sentence(text)
        return mini_text
    return text


def remove_urls(text: str):
    """
    Удаляет все URL-адреса из текста.

    :param text: Исходный текст, содержащий URL-адреса.
    :return: Текст без URL-адресов.
    """
    # Регулярное выражение для поиска URL-адресов
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Заменяем найденные URL-адреса на пустую строку
    cleaned_text = url_pattern.sub('', text)
    return cleaned_text


def remove_path(text):
    """
    Удаляет путь из текста.

    :param text: Исходный текст, содержащий путь.
    :return: Текст без пути.
    """
    # Регулярное выражение для поиска пути
    path_pattern = re.compile(r'\b(?:[A-Za-z]:)?[\\/](?:[^\\/<>\s]+[\\/])*[^\\/<>\s]+')
    # Удаляем найденный путь
    cleaned_text = path_pattern.sub('', text)
    return cleaned_text.strip()  # Убираем лишние пробелы


def find_emoticons(text: str, coefficient: float = 1.5, start_boost: float = 0.7) -> list[float]:
    """Принимает текст text и ищет смайлики. coefficient - коэффициент увеличения значения тональности по направлению,
    в котором найден эмотикон. start_boost - стартовая прибавка к значению тональности при первом обнаружении. Возвращает вектор тональности"""
    vector = [0.0, 0.0, 0.0]
    if isinstance(text, list):
        result = [find_emoticons(element, coefficient=coefficient, start_boost=start_boost) for element in text]
        for el in result:
            for i in range(3):
                vector[i] += el[i]
        for i in range(3):
            vector[i] /= len(result)
        return vector

    # проверка на смайлики
    for word in text.split():
        print(word)
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
    if sum(vector) == 0:
        # проверку на простые смайлики ')' и '('
        count_close = text.count(')')
        count_open = text.count('(')
        if count_close > count_open:
            vector[2] = start_boost
        elif count_open > count_close:
            vector[0] = start_boost
    print(vector)
    return vector


# # Тесты
# start_time = time.time()
#
#
# # dataset
# dataset = pd.read_excel('dataset_comments_35.xlsx')
# # sentiments = [get_sentiment(text, return_type='proba-label', emoji=False, start_boost=0.7, coefficient=1.5, del_name=True) for text in dataset['MessageText']] # Результат
# sentiments2 = [get_sentiment(text, return_type='proba-label', emoji=True, start_boost=0.7, coefficient=1.5, del_name=True) for text in dataset['MessageText']]
#
# print(time.time()-start_time)
# # print(sentiments)
# # dataset['Result'] = sentiments
# dataset['Result_Emoji'] = sentiments2
# # count_True = 0
# # for i in range(168):
# #     if dataset['Class'][i] == dataset['Result'][i][0][0].upper():
# #         count_True += 1
# #     else:
# #         print(dataset.iloc[i, [2, 3, 4]], '\n')
# # print('-'*150)
# # print('Result', count_True/168)
# count_True2 = 0
# for i in range(168):
#     if dataset['Class'][i] == dataset['Result_Emoji'][i][0][0].upper():
#         count_True2 += 1
#     else:
#         print(dataset.iloc[i, [2, 3, 4]], '\n')
# print('Result_Emoji', count_True2/168)
# #
# result_data = pd.DataFrame({
#     "Имя": [],
#     "Счет": []
# })
# # result_data.loc[len(result_data)] = ['Result', count_True/168]
# # result_data.loc[len(result_data)] = ['Result_Emoji', count_True2/168]
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
# for thresh in range(100, 120, 5):
#     thresh /= 100
#     sentiments = [get_sentiment(text, return_type='proba-label', emoji=True, start_boost=0.7, coefficient=1.5, del_name=True, name_thresh=thresh) for text in dataset['MessageText']]
#     count_True = 0
#     for i in range(168):
#         if dataset['Class'][i] == sentiments[i][0][0].upper():
#             count_True += 1
#     print(f'{thresh}: {count_True/168}')
#     result_data.loc[len(result_data)] = [f'{thresh}', count_True/168]
#
# result_data.to_excel('Result_data_thresh.xlsx')
# dataset.to_excel('Model1.xlsx')