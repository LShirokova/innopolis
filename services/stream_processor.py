import time
import logging
import redis
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STREAM_KEY = "sensor_stream"
GROUP_NAME = "prediction_group"

class StreamProcessor:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'), 
            port=os.getenv('REDIS_PORT', 6379), 
            decode_responses=True
        )
        # Создаем Consumer Group (аналог Kafka consumer group), если ее нет
        try:
            self.redis_client.xgroup_create(STREAM_KEY, GROUP_NAME, id='0', mkstream=True)
        except redis.exceptions.ResponseError:
            pass # Группа уже существует

    def produce_sensor_data(self, engine_id: int, sensor_data: dict):
        """Имитация отправки данных с датчиков в поток (Producer)."""
        payload = {
            "engine_id": engine_id,
            "timestamp": int(time.time()),
            **sensor_data
        }
        # Отправка в Stream
        self.redis_client.xadd(STREAM_KEY, payload)
        logger.info(f"📤 Отправлены данные двигателя {engine_id} в поток")

    def consume_and_update_features(self):
        """Чтение из потока и обновление фичей в Online Store (Consumer)."""
        logger.info("🔴 Consumer запущен. Ожидание данных из потока...")
        while True:
            # Читаем новые сообщения (блокируемся на 1 секунду, если пусто)
            messages = self.redis_client.xreadgroup(
                GROUP_NAME, "worker-1", {STREAM_KEY: '>'}, count=1, block=1000
            )
            
            if messages:
                for stream, msg_list in messages:
                    for msg_id, msg_data in msg_list:
                        engine_id = msg_data['engine_id']
                        logger.info(f"📥 Получено сообщение {msg_id} для двигателя {engine_id}")
                        
                        # В реальной системе здесь мы бы пересчитывали rolling window
                        # на основе истории, хранящейся в Redis List для этого двигателя.
                        # Для демонстрации архитектуры: симулируем обновление состояния
                        self._simulate_rolling_window_update(engine_id, msg_data)
                        
                        # Подтверждаем обработку сообщения (Ack)
                        self.redis_client.xack(STREAM_KEY, GROUP_NAME, msg_id)

    def _simulate_rolling_window_update(self, engine_id: str, new_data: dict):
        """Упрощенная логика: добавляем новое значение в 'окно' и пересчитываем среднее."""
        list_key = f"history:{engine_id}:sensor_11" # Пример для одного сенсора
        
        # Добавляем в конец списка (Ограничим окно 10 элементами для имитации 10w)
        self.redis_client.rpush(list_key, new_data.get('sensor_11', 0))
        self.redis_client.ltrim(list_key, -10, -1) # Оставляем только последние 10
        
        # Считаем новое среднее окна
        history = self.redis_client.lrange(list_key, 0, -1)
        if history:
            new_mean = np.mean([float(x) for x in history])
            # Обновляем фичу в Online Store
            self.redis_client.hset(f"features:{engine_id}", "sensor_11_mean_10w", new_mean)
            logger.info(f"🔄 Обновлена фича sensor_11_mean_10w = {new_mean:.2f} для двигателя {engine_id}")

if __name__ == "__main__":
    processor = StreamProcessor()
    
    # Демо: отправляем тестовый пакет и запускаем консьюмер
    processor.produce_sensor_data(engine_id=1, sensor_data={"sensor_11": 45.2, "sensor_4": 1200.5})
    
    # Раскомментируй для постоянного прослушивания потока:
    # processor.consume_and_update_features()