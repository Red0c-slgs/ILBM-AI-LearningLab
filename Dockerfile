FROM python:3.12.3-slim


WORKDIR /app


# Копируем файлы и директории
COPY templates ./templates
COPY requirementsdocker.txt .
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
RUN pip install -r requirementsdocker.txt
RUN mkdir "upload"
RUN mkdir "processed"

# Запускаем приложение с помощью Hypercorn
EXPOSE 80
CMD ["hypercorn", "app:app", "--bind", "0.0.0.0:80"]
