from quart import Quart, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import pandas as pd
import asyncio
import model
import model_statistics as ms
from quart import send_file
import io

app = Quart(__name__)

# Папка для загрузки файлов
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}


def allowed_file(filename):
    """Проверка расширения файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


async def data_sentiment(data_name: str):
    """Принимает название файла с расширением, обрабатывает и возвращает датасет"""
    dataset = pd.read_excel(data_name)
    if 'MessageText' not in dataset.columns:  # Если нет колонки с текстом
        raise ValueError("Файл должен содержать колонку 'MessageText'")

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


@app.route('/')
async def index():
    """Главная страница с формой загрузки файла и анализа текста"""
    return await render_template('upload.html')


@app.route('/upload', methods=['POST'])
async def upload_file():
    """Обработка загрузки файла"""
    if 'file' not in (await request.files):
        return await render_template('upload.html', error="Файл не выбран")

    file = (await request.files)['file']

    if file.filename == '':
        return await render_template('upload.html', error="Файл не выбран")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        await file.save(file_path)

        # Перенаправляем на страницу статистики с именем файла
        return redirect(url_for('show_statistics', filename=filename))
    else:
        return await render_template('upload.html', error="Недопустимый формат файла")


@app.route('/statistics')
async def show_statistics():
    # Получаем имя файла из query-параметров
    filename = request.args.get('filename')
    if not filename:
        return await render_template('upload.html', error="Файл не указан")

    # Полный путь к файлу
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        return await render_template('upload.html', error="Файл не найден")

    # Анализируем файл
    dataset = await data_sentiment(file_path)
    stats = ms.bar_chart(dataset)

    # Передаем данные в шаблон statistics.html
    return await render_template(
        'statistics.html',
        labels=stats["labels"],  # Метки для графика
        counts=stats["counts"],  # Данные для графика
        dataset=dataset,  # Данные для таблицы
        filename=filename  # Имя файла для скачивания
    )


@app.route('/download')
async def download_file():
    # Получаем имя файла из query-параметров
    filename = request.args.get('filename')
    if not filename:
        return await render_template('upload.html', error="Файл не указан")

    # Полный путь к файлу
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        return await render_template('upload.html', error="Файл не найден")

    # Анализируем файл
    dataset = await data_sentiment(file_path)

    # Сохраняем датасет в буфер
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataset.to_excel(writer, index=False)
    output.seek(0)

    # Возвращаем файл для скачивания
    return await send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        attachment_filename=f"analyzed_{filename}"
    )


@app.route('/text', methods=['POST'])
async def text_mess_get():
    """Анализ текста, введенного пользователем"""
    data = (await request.form).get("text")
    sent_dict = {"B": "Негативный", "N": "Нейтральный", "G": "Позитивный"}
    if not data:
        return await render_template('upload.html', error="Текст не предоставлен")

    try:
        sentiment = sent_dict[await text_sentiment(data)]
        return await render_template('upload.html', message=f"Тональность текста: {sentiment}",text = data)
    except Exception as e:
        return await render_template('upload.html', error=str(e))


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)