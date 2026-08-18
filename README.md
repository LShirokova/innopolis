## Структура
├── data/ # Хранилище данных
│ ├── raw/ # Сырые данные txt
├── src/ # Скрипты ETL процесса
│ ├── db_manager.py # Функции работы с БД
│ ├── etl_pipeline.py # Работа с данными, EDA, Feature Engineering, загрузка в БД
└── README.md
└── requirements.txt # Python библиотеки


## Быстрый старт


### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Подготовка и загрузка данных в БД 
```bash
python scripts/etl_pipeline.py
```