from quart import Quart, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import model_statistics as ms
import async_func as fma
from config import UPLOAD_FOLDER, PROCESSED_FOLDER, ALLOWED_EXTENSIONS
import os
import io


app = Quart(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename: str)->bool:
    """
    Проверка расширения файла.
    :param filename: Имя файла.
    :return: True, если расширение файла разрешено, иначе False.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
async def index():
    """
    Главная страница с формой загрузки файла и анализа текста.
    :return: Шаблон upload.html.
    """
    return await render_template('upload.html')


@app.route('/upload', methods=['POST'])
async def upload_file():
    """
    Обрабатывает загрузку файла.
    :return: Редирект на страницу статистики или сообщение об ошибке.
    """
    if 'file' not in (await request.files):
        return await render_template('upload.html', error="Файл не выбран")

    file = (await request.files)['file']

    # Проверка, что файл был выбран
    if file.filename == '':
        return await render_template('upload.html', error="Файл не выбран")

    # Проверка расширения файла и его сохранение
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Безопасное имя файла
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        await file.save(file_path)

        # Перенаправление на страницу статистики
        return redirect(url_for('show_statistics', filename=filename))
    else:
        return await render_template('upload.html', error="Недопустимый формат файла")


@app.route('/statistics')
async def show_statistics():
    """
    Отображает статистику по загруженному файлу.
    :return: Шаблон statistics.html с данными для отображения.
    """
    filename = request.args.get('filename')  # Получение имени файла из запроса
    if not filename:
        return await render_template('upload.html', error="Файл не указан")

    # Абсолютный путь к файлу
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Проверка существования файла
    if not os.path.exists(file_path):
        return await render_template('upload.html', error="Файл не найден")

    # Анализ тональности текстов в файле
    dataset = await fma.data_sentiment(file_path)

    # Сохранение обработанного файла
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    dataset.to_excel(processed_path, index=False)

    # Генерация статистики
    stats_chart = ms.bar_chart(dataset)  # График тональности
    stats_len = ms.text_lengths_by_tone(dataset)  # Длина текстов по тональности
    stats_chastotnost = ms.top_words_by_tone(dataset)  # Топ слов по тональности
    stats_time = ms.tone_over_time(dataset)  # Тональность по времени
    stats_user = ms.users_tone(dataset)  # Тональность пользователей

    # Отображение статистики в шаблоне
    return await render_template(
        'statistics.html',
        labels=stats_chart["labels"],  # Метки для графика
        counts=stats_chart["counts"],  # Количество
        percentages=stats_chart["percentages"],  # Проценты
        dataset=dataset,  # Данные для таблицы
        filename=filename,  # Имя файла для скачивания
        stats_len=stats_len,  # Статистика длины текстов
        stats_chastotnost=stats_chastotnost,  # Топ слов по тональности
        stats_time=stats_time,  # Тональность по времени
        stats_user=stats_user  # Тональность пользователей
    )


@app.route('/download')
async def download_file():
    """
    Позволяет скачать обработанный файл.
    :return: Файл для скачивания.
    """
    filename = request.args.get('filename')  # Получение имени файла из запроса
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    # Проверка существования файла
    if not os.path.exists(processed_path):
        return await render_template('upload.html', error="Файл не найден")

    # Чтение файла и преобразование в байтовый поток
    dataset = pd.read_excel(processed_path)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataset.to_excel(writer, index=False)
    output.seek(0)

    # Отправка файла пользователю
    return await send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        attachment_filename=f"analyzed_{filename}"
    )


@app.route('/text', methods=['POST'])
async def text_mess_get():
    """
    Анализирует текст, введенный пользователем.
    :return: Результат анализа тональности текста.
    """
    data = (await request.form).get("text")  # Получение текста из формы
    sent_dict = {"B": "Негативный", "N": "Нейтральный", "G": "Позитивный"}  # Словарь тональностей

    # Проверка наличия текста
    if not data:
        return await render_template('upload.html', error="Текст не предоставлен")

    try:
        # Анализ тональности текста
        sentiment = sent_dict[await fma.text_sentiment(data)]
        return await render_template('upload.html', message=f"Тональность текста: {sentiment}", text=data)
    except Exception as e:
        # Обработка ошибок
        return await render_template('upload.html', error=str(e))




if __name__ == '__main__':
    # Создание папок, если они не существуют
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    # Запуск приложения
    app.run(debug=True)