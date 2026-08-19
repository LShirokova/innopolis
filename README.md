## Структура
├── data/ # Хранилище данных
│ ├── raw/ # Сырые данные txt
├── docs/ # Сохраненные графики
├── models/ # Обученные модели
├── src/ # Скрипты ETL процесса
│ ├── db_manager.py # Функции работы с БД
│ ├── etl_pipeline.py # Работа с данными, EDA, Feature Engineering, загрузка в БД
│ ├── train.py # Обучение моделей
└── README.md
└── requirements.txt # Python библиотеки


## Быстрый старт

### 1. Установка библиотек
```bash
pip install -r requirements.txt
```

### 2. Подготовка и загрузка данных в БД 
```bash
python src/etl_pipeline.py
```

### 3. Обучение
```bash
python src/train.py
```