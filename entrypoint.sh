#!/bin/bash
set -e # Останавливать скрипт при любой ошибке

echo "========================================"
echo "🚀 ЗАПУСК СИСТЕМЫ PREDICTIVE MAINTENANCE"
echo "========================================"

# 1. Ждем готовности PostgreSQL и Redis
echo "⏳ Ожидание запуска PostgreSQL и Redis..."
python -c "
import time, psycopg2, redis
while True:
    try:
        psycopg2.connect(dbname='mlops_db', user='admin', password='secret', host='postgres')
        redis.Redis(host='redis', port=6379).ping()
        break
    except Exception:
        time.sleep(1)
print('✅ Базы данных готовы к работе!')
"

# 2. Запускаем ETL
echo "📊 ЗАПУСК ETL ПАЙПЛАЙНА..."
python -m src.etl_pipeline

# 3. Обучаем модель и сохраняем артефакт
echo "🧠 ОБУЧЕНИЕ МОДЕЛИ..."
python -m src.train

# 4. Синхронизируем фичи в Online Store (Redis)
echo "📦 СИНХРОНИЗАЦИЯ FEATURE STORE..."
python -m services.feature_store

echo "========================================"
echo "✅ ВСЕ СЕРВИСЫ ИНИЦИАЛИЗИРОВАННЫЫ"
echo "🌐 ЗАПУСК FASTAPI НА ПОРТУ 8000..."
echo "========================================"

# 5. Запускаем API сервер (exec заменяет текущий процесс bash на uvicorn)
exec uvicorn api.main:app --host 0.0.0.0 --port 8000