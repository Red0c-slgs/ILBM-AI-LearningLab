from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import re
import logging
import pandas as pd
import model


app = Flask(__name__)

# Папка для загрузки файлов
# Добавить проверку на наличие свободной директории в папке
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'xlsx','csv'}
#мб сделать распрасивание для cvs и ods


def data_sentiment(data_name: str):
    """Принимает название файла с расширением (и относительным путем), обрабатывает и возвращает датасет"""
    dataset = pd.read_excel(data_name)
    if 'MessageText' not in dataset.columns:  # Если нет колонки с текстом
        # Надо как-то обработать
        pass
    sentiments = [model.get_sentiment(text, return_type='score-label', emoji=True, del_name=True)
                  for text in dataset['MessageText']]
    dataset['Sentiment'] = sentiments
    return dataset


def text_sentiment(text: str):
    """Принимает текст, обрабатывает и возвращает метку-результат"""
    return model.get_sentiment(text, return_type='score-label', emoji=True, del_name=True)


# Проверка расширения файла
# оформить регулярку
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('upload.html')

#написать асинхронный интерфейс, мб сформировать очередь, а обрабатывать отдельно.
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('upload.html', error="Файл не выбран")

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', error="Файл не выбран")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('upload.html', message="Файл успешно загружен")
    else:
        return render_template('upload.html', error="Недопустимый формат файла")


@app.route('/text', methods=['POST'])
def text_mess_get():
    # Получаем данные из запроса
    data = request.form.get("text")
    sent_dict = {"B":"Негативный", "N":"Нейтральный","G":"Позитивный"}
    if not data:
        return render_template('upload.html', error="Текст не предоставлен")
    sentiment = sent_dict[text_sentiment(data)]
    return render_template('upload.html', message=f"Тональность текста: {sentiment}")



if __name__ == '__main__':
    app.run(debug=True,threaded=True)