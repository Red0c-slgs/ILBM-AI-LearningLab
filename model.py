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
        proba = np.ndarray([0, 0, 0])
        for score in results:
            proba += score
        proba /= np.ndarray([n, n, n])
    else: proba = analysis(data)
    return proba


def get_sentiment(text: str, return_type: str = 'label', passing_threshold: float = 1/3) -> str | tuple[str, float] | np.ndarray:
    """Определяет тональность текста text. return_type может быть: 'label' - наибольшая метка, 'score' - числовая оценка,
    'score-label' - числовая оценка, переведенная в метку, 'proba' - вероятности для каждой метки"""
    # обработка текста
    data = preprocessing(text)
    proba = get_proba(data)
    # Выбор типа возвращаемых данных
    lbl = ['B', 'N', 'G']
    if return_type == 'label':
        return lbl[proba.argmax()]
    if return_type == 'score':
        return proba.dot([-1, 0, 1])
    if return_type == 'score-label':
        score = proba.dot([-1, 0, 1])
        if score > passing_threshold:
            return 'G', score
        if score > - passing_threshold:
            return 'N', score
        return 'B', score
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
    for sentence in re.split(r'(?<=[.!?…]) ', text):
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


# # Тесты
# start_time = time.time()
#
#
# # dataset
# dataset = pd.read_excel('dataset_comments_35.xlsx')
# dataset['MessageText'] = dataset['MessageText'].apply(clean_html)
# sentiments = [get_sentiment(text, return_type='score-label') for text in dataset['MessageText']] # Результат
#
# print(time.time()-start_time)
# print(sentiments)
#
# count_True = 0
# for i in range(85):
#     if dataset['Class'][i] == sentiments[i][0].upper():
#         count_True += 1
# print(count_True/85)
# dataset['Result'] = sentiments
#
# dataset.to_excel('Model1.xlsx')