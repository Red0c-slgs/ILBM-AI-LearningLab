from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import re
import logging


app = Flask(__name__)

# Папка для загрузки файлов
# Добавить проверку на наличие свободной директории в папке
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'xlsx'}
#мб сделать распрасивание для cvs и ods

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

if __name__ == '__main__':
    # Создаем папку для загрузок, если её нет
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)