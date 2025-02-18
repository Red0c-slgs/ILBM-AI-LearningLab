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

# Инициализация анализатора
morph = pymorphy3.MorphAnalyzer()

# Подготовка модели
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=tokenizer)


def upload_smiles(name: str) -> set[str]:
    """
    Загрузка списка эмотикоинов из файла.
    :param name: Имя файла.
    :return: Set из эмотикоинов.
    """
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
    """
    Анализирует текст и возвращает все вероятности
    :param text: Исходный текст.
    :return: Вектор тональности.
    """
    with torch.no_grad():
        proba = sentiment_task(text, return_all_scores=True)
    return np.array([vec['score'] for vec in proba[0]])


def get_proba(data: str | list[str]) -> np.ndarray:
    """
    В зависимости от входных данных возвращает результат анализа текста
    :param data: Исходный текст, либо список текстов (для слишком больших текстов).
    :return: Вектор тональности.
    """
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


def get_sentiment(text: str, return_type: str = 'score-label', passing_threshold: float = 1/3, emoji: bool = True,
                  coefficient: float = 1.5, start_boost: float = 0.7, del_name: bool = True, name_thresh: float = 0.75,
                  del_html: bool = True, del_url: bool = True, del_path: bool = True, check_solo_emoji: bool = True) \
        -> (str | tuple[str, float] | np.ndarray | tuple[str, np.ndarray]):
    """
    Определяет тональность текста.

    :param text: Исходный текст.
    :param return_type: Тип возвращаемого значения. Возможные значения:
        - 'proba': Вектор тональности.
        - 'label': Метка наибольшей вероятности.
        - 'score': Числовая оценка тональности (Разность позитива и негатива).
        - 'score-label': Числовая оценка 'score', переведенная в буквенную метку.
        - 'score-label-proba': Числовая оценка 'score', переведенная в буквенную метку и вектор тональности.
        По умолчанию: 'score-label'.
    :param passing_threshold: Порог определения полярной тональности (0 отключает нейтральные метки)
    :param emoji: Флаг поиска и учета эмотиконов.
    :param coefficient: Коэффициент умножения при обнаружении эмотикоина.
    :param start_boost: Стартовая прибавка при обнаружении первого эмотикоина.
    :param del_name: Флаг удаления имен из текста.
    :param name_thresh: Порог определения имен для удаления.
    :param del_html: Флаг удаления HTML-разметки.
    :param del_url: Флаг удаления URL-ссылок.
    :param del_path: Флаг удаления путей.
    :param check_solo_emoji: Проверка является ли текст одиночным эмотикоином.

    :return: Метка или вектор тональности.
    """
    #print(f'Начата обработка текста {text[:30]}')
    # обработка текста
    data = preprocessing(text, del_name=del_name, name_thresh=name_thresh, del_html=del_html, del_url=del_url,
                         del_path=del_path)
    # Проверка соло слова
    if isinstance(data, str) and len(data.split()) == 1:
        # print(data)
        if bool(re.match(r'[a-zA-Zа-яА-Я]', data)):
            proba = np.array([0.0, 0.0, 0.0])
        if check_solo_emoji:
            if data.split()[0] in positive_emoticons:
                proba = np.array([0.0, 0.0, 0.7])
            elif data.split()[0] in negative_emoticons:
                proba = np.array([0.7, 0.0, 0.0])
            elif 'proba' not in locals():
                proba = get_proba(data)
        # print(proba)
    else:
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
    if return_type == 'score-label-proba':
        score = proba.dot([-1, 0, 1])
        if score > passing_threshold:
            return 'G', proba
        if score > - passing_threshold:
            return 'N', proba
        return 'B', proba
    return proba


def clean_html(text: str) -> str | None:
    """
    Очищает текст text от HTML-разметки
    :param text: Исходный текст.
    :return: Текст без HTML-разметки.
    """
    if text is None or pd.isna(text):  # Проверка на NaN
        return text
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def split_sentence(text: str, window_size: int = 514) -> list[str]:
    """
    Разбивает большой текст на предложения
    :param text: Исходный текст.
    :param window_size: Максимальный размер текста, к которому нужно преобразовать.
    :return: Список меньших текстов.
    """
    sentences = []  # список предложений
    for sentence in re.split(r"(?<=[.!?…]) ", text):
        if len(sentence) > window_size: # если предложение больше окна
            short_sentences = []
            words = sentence.split()
            count = 0
            while count < len(words):
                short_sentence = ''
                while count < len(words) and len(short_sentence + ' ' + words[count]) < window_size:
                    short_sentence += ' ' + words[count] if short_sentence else words[count]
                    count += 1
                short_sentences.append(short_sentence)
            sentences.extend(short_sentences)
        else:
            sentences.append(sentence)
    return sentences


def delete_name(text: str, prob_thresh: float = 0.75) -> str:
    """
    Удаляет имена
    :param text: Исходный текст.
    :param prob_thresh: Порог определения имен для удаления.
    :return: Текст без имен.
    """
    for word in nltk.word_tokenize(text):
        for p in morph.parse(word):
            if 'Name' in p.tag and p.score >= prob_thresh:
                text = text.replace(word, '')
    # Удаление лишних пробелов
    text = ' '.join(text.split())
    return text


def preprocessing(text: str, window_size: int = 514, del_name: bool = False, name_thresh: float = 0.75,
                  del_html: bool = True, del_url: bool = True, del_path: bool = True) -> str | list[str]:
    """
    Предварительная обработка текстов
    :param text: Исходный текст.
    :param window_size: Максимальный размер текста, к которому нужно преобразовать.
    :param del_name: Флаг удаления имен.
    :param name_thresh: Порог определения имен для удаления.
    :param del_html: Флаг удаления HTML-разметки.
    :param del_url: Флаг удаления URL-ссылок.
    :param del_path: Флаг удаления путей.
    :return: Обработанный текст, либо список текстов.
    """
    if del_html:
        text = clean_html(text)
    if del_url:
        text = remove_urls(text)
    if del_path:
        text = remove_path(text)
    if del_name:
        text = delete_name(text, prob_thresh=name_thresh)
    if len(text) > window_size:
        mini_text = split_sentence(text)
        return mini_text
    return text


def remove_urls(text: str):
    """
    Удаляет все URL-адреса из текста.
    :param text: Исходный текст.
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
    :param text: Исходный текст.
    :return: Текст без пути.
    """
    # Регулярное выражение для поиска пути
    path_pattern = re.compile(r'\b(?:[A-Za-z]:)?[\\/](?:[^\\/<>\s]+[\\/])*[^\\/<>\s]+')
    # Удаляем найденный путь
    cleaned_text = path_pattern.sub('', text)
    return cleaned_text.strip()  # Убираем лишние пробелы


def find_emoticons(text: str | list[str], coefficient: float = 1.5, start_boost: float = 0.7) -> list[float]:
    """
    Принимает текст и ищет смайлики.
    :param text: Исходный текст.
    :param coefficient: Коэффициент увеличения значения тональности по направлению, в котором найден эмотикон.
    :param start_boost: Стартовая прибавка к значению тональности при первом обнаружении.
    :return: Вектор тональности.
    """
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
    return vector

