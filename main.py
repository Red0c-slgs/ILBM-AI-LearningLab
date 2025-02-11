# !pip install transformers sentencepiece --quiet
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from bs4 import BeautifulSoup


model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()


def get_sentiment(text, return_type='label'):
    """ Calculate sentiment of a text. `return_type` can be 'label', 'score' or 'proba' """
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
    if return_type == 'label':
        return model.config.id2label[proba.argmax()]
    if return_type == 'score':
        return proba.dot([-1, 0, 1])
    return proba

def clean_html(text):
    if pd.isna(text):  # Проверка на NaN
        return text
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


start_time = time.time()


# dataset
dataset = pd.read_excel('dataset_comments_500/dataset_comments.xlsx')
dataset['MessageText'] = dataset['MessageText'].apply(clean_html)
sentiments = [get_sentiment(text) for text in dataset['MessageText']]

print(time.time()-start_time)