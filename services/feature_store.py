import os
import logging
import pandas as pd
import redis
from dotenv import load_dotenv
from src.db_manager import DatabaseManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureStore:
    def __init__(self):
        # Инициализация Offline Store (PostgreSQL)
        self.db = DatabaseManager(
            db_user=os.getenv('DB_USER'), db_pass=os.getenv('DB_PASSWORD'),
            db_host=os.getenv('DB_HOST'), db_port=os.getenv('DB_PORT'),
            db_name=os.getenv('DB_NAME')
        )
        # Инициализация Online Store (Redis)
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'), 
            port=os.getenv('REDIS_PORT', 6379), 
            decode_responses=True
        )

    def sync_features_to_online_store(self):
        """
        Материализует последние актуальные фичи из Offline (Postgres) в Online (Redis).
        В реальном проде это делает Feast или специализированный Feature Server.
        """
        logger.info("🟢 Синхронизация фичей: Postgres (Offline) -> Redis (Online)...")
        
        # Читаем фичи из Offline Feature Store
        df = pd.read_sql("SELECT * FROM features.ml_features", con=self.db.engine)
        
        # Для каждого двигателя берем ПОСЛЕДНЮЮ строку (актуальное состояние)
        latest_features = df.sort_values('cycle').groupby('engine_id').last().reset_index()
        feature_cols = [c for c in latest_features.columns if c not in ['engine_id', 'cycle', 'target_label']]
        
        loaded_count = 0
        for _, row in latest_features.iterrows():
            engine_id = str(row['engine_id'])
            # Сохраняем фичи как Hash в Redis (ключ: features:engine_id)
            feature_dict = row[feature_cols].to_dict()
            
            # Преобразуем numpy типы в нативные Python для Redis
            feature_dict = {k: float(v) if pd.notnull(v) else None for k, v in feature_dict.items()}
            
            self.redis_client.hset(f"features:{engine_id}", mapping=feature_dict)
            loaded_count += 1
            
        logger.info(f"✅ Загружено {loaded_count} записей в Online Feature Store (Redis)")

if __name__ == "__main__":
    fs = FeatureStore()
    fs.sync_features_to_online_store()