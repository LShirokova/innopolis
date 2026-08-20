# Используем slim образ для быстрого старта
FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости для psycopg2
RUN apt-get update && apt-get install -y libpq-dev gcc

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и обученную модель
COPY . .

# Запускаем Redis и API сервис
RUN chmod +x entrypoint.sh

# Пробрасываем порт для FastAPI
EXPOSE 8000

# Запускаем API сервер
ENTRYPOINT ["./entrypoint.sh"]