## Структура
├── data/ # Хранилище данных
│ ├── raw/ # Сырые данные txt
├── docs/ # Сохраненные графики, презентация
├── models/ # Обученные модели
├── services/ # Скрипты потоковой обработки
│ ├── feature_store.py #
│ ├── stream_processor.py #
├── src/ # Скрипты ETL процесса
│ ├── db_manager.py # Функции работы с БД
│ ├── etl_pipeline.py # Работа с данными, EDA, Feature Engineering, загрузка в БД
│ ├── train.py # Обучение моделей
└── Dockerfile # Конфигурация Docker
└── docker-compose.yml # Конфиг контейнера
└── README.md
└── requirements.txt # Python библиотеки


## Быстрый старт (без Docker)

### 1. Клонировать репозиторий репозиторий:
```bash
git clone https://github.com/LShirokova/innopolis.git
```

### 2. Установка библиотек
```bash
pip install -r requirements.txt
```

### 3. Подготовка и загрузка данных в БД 
```bash
python src/etl_pipeline.py
```

### 4. Обучение
```bash
python src/train.py
```


## Запуск с Docker
### 1. Поднять докер со всей инфраструктурой
```
docker-compose up --build
```
Докер с зависимостями поднимается за 14 шагов.
Далее происходит ETL, обучение и сохранение модели
### 2. Сервис доступен по адресу:
http://localhost:8000/docs