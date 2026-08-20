## Структура проекта
```
├── api # FastApi
├── data/ # Хранилище данных
│ ├── raw/ # Сырые данные txt
├── docs/ # Сохраненные графики, презентация
├── models/ # Обученные модели
├── services/ # Скрипты потоковой обработки
│ ├── feature_store.py # Для Redis
│ ├── stream_consumer.py # Слушатель потока
│ ├── stream_producer.py # Потоковый Producer отправляет данные в Consumer
├── src/ # Скрипты ETL процесса
│ ├── db_manager.py # Функции работы с БД
│ ├── etl_pipeline.py # Работа с данными, EDA, Feature Engineering, загрузка в БД
│ ├── train.py # Обучение моделей
└── .dockerignore # Список исключений для Docker
└── .gitignore # Список исключений для GIT
└── docker-compose.yml # Конфиг контейнера
└── Dockerfile # Конфигурация Docker
└── entrypoint.sh # Bash скрипт для запуска программы
└── README.md
└── requirements.txt # Python библиотеки
```


## Быстрый старт (без Docker)

### 1. Клонировать репозиторий репозиторий:
```
git clone https://github.com/LShirokova/innopolis.git
```

### 2. Установка библиотек
```
pip install -r requirements.txt
```

### 3. Подготовка и загрузка данных в БД 
```
python src/etl_pipeline.py
```

### 4. Обучение
```
python src/train.py
```
### 5. Синхронизировать фичи
```
python -m services.feature_store
```
### 6. Запустить Redis
```
docker-compose up -d redis
```
### 7. Запуск
```
uvicorn api.main:app --reload
```
Сервис откроется по адресу: http://localhost:8000/docs

---

## Запуск с Docker
### 1. Поднять докер со всей инфраструктурой
```
docker-compose up --build
```
Докер с зависимостями поднимается за 14 шагов.  
Если модель уже обучена, переходи к шагу 2.

### 1.1 Если модель не обучена, создать базу и залить данные
docker-compose exec ml_api python -m src.etl_pipeline

### 1.2 Обучить модель (сохранится прямо в контейнер)
docker-compose exec ml_api python -m src.train

### 1.3 Перезапустить контейнер, чтобы API подхватил новую модель
docker-compose restart ml_api

## 2. Сервис доступен по адресу:
http://localhost:8000/docs

---

# Потоковая обработка
### 1. Запустить Redis
```
docker-compose up -d redis
```
### 2. Залить фичи
```
python -m services.feature_store
```
### 3. Запустить API
```
uvicorn api.main:app --reload
```
### 4. Запустить Producer\Consumer
**В терминале 1:**  
```
python -m services.stream_consumer
```
**В терминале 2:**  
```
python -m services.stream_producer
```
### 🟢 Enjoy!