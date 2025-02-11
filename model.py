import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # токенизация
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint) # загрузка модели


def get_sentiment(text: str, return_type: str = 'label') -> str | float | np.ndarray:
    """Определяет тональность текста text. return_type может быть: 'label' - метка, 'score' - числовая оценка,
    'proba' - вероятности для каждой метки"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
    if return_type == 'label':
        return model.config.id2label[proba.argmax()]
    if return_type == 'score':
        return proba.dot([-1, 0, 1])
    return proba


def clean_html(text: str) -> str:
    """Очищает текст text от HTML-разметки"""
    if pd.isna(text):  # Проверка на NaN
        return text
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


start_time = time.time()


# dataset
dataset = pd.read_excel('dataset_comments_500/dataset_comments.xlsx')
dataset['MessageText'] = dataset['MessageText'].apply(clean_html)
sentiments = [get_sentiment(text) for text in dataset['MessageText']] # Результат

print(time.time()-start_time)