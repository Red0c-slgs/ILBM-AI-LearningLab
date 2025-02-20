from functools import lru_cache
import model
import time
import asyncio
import pandas as pd


@lru_cache(maxsize=1000)  # Кэширует до 1000 результатов
def cached_get_sentiment(text):
    """
    Кэшированная версия функции model.get_sentiment.
    :param text: Текст для анализа.
    :return: Результат анализа тональности.
    """
    return model.get_sentiment(
        text, return_type='score-label', passing_threshold=0.4, coefficient=1.2,
        start_boost=1, name_thresh=0.2
    )

async def data_sentiment(data_name: str):
    """
    Анализирует тональность текстов в файле.
    :param data_name: Имя файла с расширением.
    :return: Датасет с добавленной колонкой 'Sentiment'.
    """
    st = time.time()
    dataset = pd.read_excel(data_name)

    if 'MessageText' not in dataset.columns:
        raise ValueError("Файл должен содержать колонку 'MessageText'")

    # Ограничение на количество одновременно выполняемых задач
    semaphore = asyncio.Semaphore(12)  # Максимум 12 задач одновременно

    async def process_text(text):
        """
        Анализирует текст с использованием кэшированной функции.
        :param text: Текст для анализа.
        :return: Результат анализа тональности.
        """
        async with semaphore:
            return await asyncio.to_thread(cached_get_sentiment, text)

    # Создаем список задач
    tasks = [process_text(text) for text in dataset['MessageText']]

    # Параллельное выполнение задач
    sentiments = await asyncio.gather(*tasks)

    dataset['Sentiment'] = sentiments
    print(f"Время обработки файла: {time.time() - st}")
    return dataset



async def text_sentiment(text: str):
    """
    Анализирует тональность одного текста.
    :param text: Текст для анализа.
    :return: Результат анализа тональности.
    """
    return await asyncio.to_thread(
        model.get_sentiment, text, return_type='score-label', passing_threshold=0.4, coefficient=1.2,
        start_boost=1, name_thresh=0.2
    )