from quart import Quart, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import model
import asyncio

app = Quart(__name__)

# Папка для загрузки файлов
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}


def data_statistics(dataset: pd.DataFrame):
    """Принимает датасет и возвращает json данные для столбчатой диаграммы"""
    labels = ['B', 'N', 'G']
    sentiment_counts = dataset['Sentiment'].value_counts()  # Подсчет количества каждого элемента
    sentiment_counts = sentiment_counts.reindex(labels, fill_value=0)  # Убедимся, что все метки присутствуют
    counts_list = sentiment_counts[labels].tolist()  # Преобразуем в список
    return {"labels": labels, "counts": counts_list}


async def data_sentiment(data_name: str):
    """Принимает название файла с расширением, обрабатывает и возвращает датасет"""
    dataset = pd.read_excel(data_name)
    if 'MessageText' not in dataset.columns:  # Если нет колонки с текстом
        # Обработка ошибки
        pass
    sentiments = []
    for text in dataset['MessageText']:
        sentiment = await asyncio.to_thread(
            model.get_sentiment, text, return_type='score-label', emoji=True, del_name=True
        )
        sentiments.append(sentiment)
    dataset['Sentiment'] = sentiments
    return dataset


async def text_sentiment(text: str):
    """Принимает текст, обрабатывает и возвращает метку-результат"""
    return await asyncio.to_thread(
        model.get_sentiment, text, return_type='score-label', emoji=True, del_name=True
    )


def allowed_file(filename):
    """Проверка расширения файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
async def index():
    return await render_template('upload.html')


@app.route('/upload', methods=['POST'])
async def upload_file():
    if 'file' not in (await request.files):
        return await render_template('upload.html', error="Файл не выбран")

    file = (await request.files)['file']

    if file.filename == '':
        return await render_template('upload.html', error="Файл не выбран")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        await file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return await render_template('upload.html', message="Файл успешно загружен")
    else:
        return await render_template('upload.html', error="Недопустимый формат файла")


@app.route('/text', methods=['POST'])
async def text_mess_get():
    # Получаем данные из запроса
    data = (await request.form).get("text")
    sent_dict = {"B": "Негативный", "N": "Нейтральный", "G": "Позитивный"}
    if not data:
        return await render_template('upload.html', error="Текст не предоставлен")
    sentiment = sent_dict[await text_sentiment(data)]
    return await render_template('upload.html', message=f"Тональность текста: {sentiment}")


if __name__ == '__main__':
    app.run(debug=True)