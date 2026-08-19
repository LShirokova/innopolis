import pandas as pd
from sqlalchemy import create_engine, inspect, text
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DatabaseManager:
    """
    Проверяет наличие БД и создает базу, если ее нет.
    Создает таблицы на основе DataFrame.
    """
    def __init__(self, db_user: str, db_pass: str, db_host: str, db_port: str, db_name: str):
        self.db_user = db_user
        self.db_pass = db_pass
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.engine = None
        
        self._ensure_database_exists()
        self.engine = create_engine(self._get_url(self.db_name))
        self.inspector = inspect(self.engine)

    def _get_url(self, db_name: str) -> str:
        return f"postgresql+psycopg2://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{db_name}"

    def _ensure_database_exists(self):
        """Подключается к служебной БД 'postgres' и создает целевую, если её нет"""
        logger.info(f"🟢 Проверка существования базы данных '{self.db_name}'...")
        
        # Подключаемся к дефолтной базе 'postgres'
        sys_engine = create_engine(self._get_url('postgres'), isolation_level="AUTOCOMMIT")
        
        try:
            with sys_engine.connect() as conn:
                # Проверяем наличие БД в системном каталоге
                result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{self.db_name}'"))
                db_exists = result.scalar() is not None
                
                if not db_exists:
                    logger.info(f"База данных '{self.db_name}' не найдена. Создание...")
                    conn.execute(text(f"CREATE DATABASE {self.db_name}"))
                    logger.info(f"✅ База данных '{self.db_name}' успешно создана.")
                else:
                    logger.info(f"⚠️ База данных '{self.db_name}' уже существует. Подключение...")
        finally:
            sys_engine.dispose()

    def init_schemas(self):
        with self.engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS cleaned"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS features"))
            conn.commit()
        logger.info("✅ Схемы БД (raw, cleaned, features) проверены/созданы.")

    def create_table_from_df(self, df: pd.DataFrame, table_name: str, schema: str):
        """Динамически создает таблицу на основе DataFrame"""
        if not self.inspector.has_table(table_name, schema=schema):
            df.head(0).to_sql(table_name, self.engine, schema=schema, index=False, if_exists='fail')
            logger.info(f"✅ Создана таблица: {schema}.{table_name} ({len(df.columns)} колонок)")
        else:
            logger.info(f"⚠️ Таблица {schema}.{table_name} уже существует. Пропуск создания.")

    def drop_table(self):
        with self.engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE raw.sensor_data"))
            conn.execute(text("TRUNCATE TABLE cleaned.sensor_data"))
            conn.execute(text("TRUNCATE TABLE features.ml_features"))
            conn.commit()
        logger.info("⚠️ База данных очищена")