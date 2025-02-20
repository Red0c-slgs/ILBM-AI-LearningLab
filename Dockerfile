FROM python:3.12.3-slim


WORKDIR /app

# Копируем файлы и директории
COPY templates ./templates
COPY requirements.txt .
COPY async_func.py .
COPY app.py .
COPY config.py .
COPY emoji_negative.txt .
COPY emoji_positive.txt .
COPY model.py .
COPY model_statistics.py .

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip и устанавливаем зависимости Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Запускаем приложение с помощью Hypercorn
CMD ["hypercorn", "app:app", "--bind", "0.0.0.0:8000"]