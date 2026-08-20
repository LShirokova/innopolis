import os
import pandas as pd
import logging
from dotenv import load_dotenv
from sqlalchemy import text
from src.db_manager import DatabaseManager

load_dotenv()

# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ENGINE_ID_COL = 'engine_id'
CYCLE_COL = 'cycle'
TARGET_COL = 'target_label'
RUL_THRESHOLD = 7 # Предсказание на 7 циклов вперед

# Настройки источника
RAW_DATA_PATH = 'data/raw/train_FD001.txt'
RAW_COLUMNS = [ENGINE_ID_COL, CYCLE_COL, 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)] # 21 сенсор

DB = DatabaseManager(
    db_user=os.getenv('DB_USER'),
    db_pass=os.getenv('DB_PASSWORD'),
    db_host=os.getenv('DB_HOST'),
    db_port=os.getenv('DB_PORT'),
    db_name=os.getenv('DB_NAME'),
)

# Инициализация схем
DB.init_schemas()


def load_to_raw_layer():
    """
    Создает схему БД.
    Загружает исходный файл в сырой слой без изменений.
    """
    logger.info("\n--- 🟢 СТАРТ: Загрузка RAW LAYER ---")

    # В датасете данные разделены пробелом
    df_raw = pd.read_csv(RAW_DATA_PATH, sep='\s+', header=None, names=RAW_COLUMNS)

    # Создаем таблицы в БД для записи сырых данных
    DB.create_table_from_df(df_raw, 'sensor_data', 'raw')

    # Пишем данные (если таблица пустая)
    with DB.engine.connect() as conn:
        count = pd.read_sql("SELECT COUNT(*) FROM raw.sensor_data", con=conn).iloc[0, 0]
        if count == 0:
            df_raw.to_sql('sensor_data', DB.engine, schema='raw', if_exists='append', index=False, chunksize=10000)
            logger.info(f"✅ RAW LAYER загружен: {df_raw.shape[0]} строк.")
        else:
            logger.info(f"⚠️ RAW LAYER уже содержит {count} строк. Загрузка пропущена.")

def load_to_cleaned_layer():
    """
        Очищает данные, формирует бизнес-таргет, удаляет константы и утечки.
    """
    logger.info("\n--- 🟢 СТАРТ: Обработка CLEANED LAYER ---")

    FLOAT_NOISE_VARIANCE_THRESHOLD = 1e-5 #  порог отсечения шумов

    # копия датафрейма
    df_clean = pd.read_sql("SELECT * FROM raw.sensor_data", con=DB.engine)

    # Находим максимальный цикл для каждого двигателя (это момент поломки)
    max_cycles = df_clean.groupby(ENGINE_ID_COL)[CYCLE_COL].max().reset_index()
    max_cycles.columns = [ENGINE_ID_COL, 'max_cycle']

    # Формализация целевой переменной (RUL)
    df_clean = df_clean.merge(max_cycles, on=ENGINE_ID_COL)
    df_clean['rul'] = df_clean['max_cycle'] - df_clean[CYCLE_COL]
    df_clean[TARGET_COL] = (df_clean['rul'] <= RUL_THRESHOLD).astype(int)

    # Удаление временных колонок, чтобы не было утечки
    df_clean.drop(columns=['max_cycle', 'rul'], inplace=True)

    # Считаем дисперсию в числовых колонках
    sensor_cols = [col for col in df_clean.columns if 'sensor_' in col]
    variances = df_clean[sensor_cols].var()

    # Находим те, где дисперсия ниже порога шума
    noisy_const_cols = variances[variances < FLOAT_NOISE_VARIANCE_THRESHOLD].index.tolist()
    cols_to_drop = [col for col in noisy_const_cols]
    logger.info(f"🎯 Обнаружены константы/шум (variance < {FLOAT_NOISE_VARIANCE_THRESHOLD}): {cols_to_drop}")

    # Удаляем их из датафрейма перед записью в Cleaned Layer
    df_clean.drop(columns=cols_to_drop, inplace=True)

    # Удаление дубликатов и обработка пропусков (стандартный ETL)
    df_clean.drop_duplicates(inplace=True)
    df_clean.ffill()

    # Создаем таблицу cleaned.sensor_data динамически под новый набор колонок
    DB.create_table_from_df(df_clean, 'sensor_data', 'cleaned')
    df_clean.to_sql('sensor_data', DB.engine, schema='cleaned', if_exists='replace', index=False, chunksize=10000)
    logger.info(f"✅ CLEANED LAYER сохранен. Распределение классов:\n{df_clean[TARGET_COL].value_counts().to_dict()}")

    return df_clean

def build_features_layer():
    """
        Для каждого сенсора генерируем агрегаты (среднее значение и стандартное отклонение) в 5 и 10 циклов.
        .rolling(window=w) - создаем скользящее окно размером w
        .reset_index(0, drop=True) - выравнивает нумерацию строк после .rolling()
    """
    logger.info("\n--- 🟢 СТАРТ: Формирование FEATURES LAYER ---")

    df_clean = pd.read_sql("SELECT * FROM cleaned.sensor_data", con=DB.engine)

    df_features = df_clean[[ENGINE_ID_COL, CYCLE_COL, TARGET_COL]].copy()

    active_sensors = [col for col in df_clean.columns if 'sensor_' in col]
    windows = [5, 10]

    # Генерируем фичи (окна)
    for sensor in active_sensors:
        for w in windows:
            # Скользящее среднее (тренд)
            df_features[f'{sensor}_mean_{w}w'] = (
                df_clean.groupby(ENGINE_ID_COL)[sensor]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            # Скользящее отклонение (стабильность/вибрация)
            df_features[f'{sensor}_std_{w}w'] = (
                df_clean.groupby(ENGINE_ID_COL)[sensor]
                .rolling(window=w, min_periods=1)
                .std()
                .fillna(0) # На первом цикле std не посчитать
                .reset_index(0, drop=True)
            )

    # Заполняем NaN на границах окон (если есть)
    df_features.ffill()

    # Создаем таблицу features.ml_features ДИНАМИЧЕСКИ (она будет содержать все сгенерированные _mean_ и _std_ колонки)
    DB.create_table_from_df(df_features, 'ml_features', 'features')

    df_features.to_sql('ml_features', DB.engine, schema='features', if_exists='replace', index=False, chunksize=10000)

    total_features = df_features.shape[1] - 3 # здесь 3 - кол-во начальных колонок [ENGINE_ID_COL, CYCLE_COL, TARGET_COL]
    logger.info(f"✅ FEATURES LAYER сохранен. Сгенерировано {total_features} ML-признаков.")

def clear_layers():
    """Очищает все слои перед загрузкой новых данных."""
    logger.info("Очистка слоев данных...")
    with DB.engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS raw.sensor_data"))
        conn.execute(text("DROP TABLE IF EXISTS cleaned.sensor_data"))
        conn.execute(text("DROP TABLE IF EXISTS features.ml_features"))
        conn.commit()
    logger.info("✅ БД: слои очищены (готовы к созданию).")


# ==========================================
# ЗАПУСК ПАЙПЛАЙНА
# ==========================================

if __name__ == "__main__":
    # При запуске пайплайна очищаем БД
    clear_layers()

    # Последовательный запуск слоев
    load_to_raw_layer()
    load_to_cleaned_layer()
    build_features_layer()
    
    logger.info("--- ✅ ETL PIPELINE УСПЕШНО ЗАВЕРШЕН ---")