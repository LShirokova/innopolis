import joblib
import redis
import pandas as pd
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

STREAM_NAME = "raw_sensor_stream"
GROUP_NAME = "realtime_ml_group"

# Сенсоры, которые не были удалены как шум в ETL (из train.py)
ACTIVE_SENSORS = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]

def calculate_features(buffer_df):
    """Динамический расчет скользящих окон"""
    features = {}
    for s in ACTIVE_SENSORS:
        # Окно 5
        mean_5 = buffer_df[s].tail(5).mean()
        std_5 = buffer_df[s].tail(5).std()
        
        features[f'{s}_mean_5w'] = 0.0 if pd.isna(mean_5) else float(mean_5)
        features[f'{s}_std_5w'] = 0.0 if pd.isna(std_5) else float(std_5)
        
        # Окно 10
        if len(buffer_df) >= 10:
            mean_10 = buffer_df[s].tail(10).mean()
            std_10 = buffer_df[s].tail(10).std()
        else:
            mean_10 = mean_5
            std_10 = 0.0
            
        features[f'{s}_mean_10w'] = 0.0 if pd.isna(mean_10) else float(mean_10)
        features[f'{s}_std_10w'] = 0.0 if pd.isna(std_10) else float(std_10)
            
    return features

def run_consumer():
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
    model = joblib.load("models/best_model.pkl")
    
    try:
        r.xgroup_create(STREAM_NAME, GROUP_NAME, id='0', mkstream=True)
    except redis.exceptions.ResponseError:
        pass

    # ВАЖНО: In-memory хранилище истории (Stateful Streaming)
    # Формат: { "engine_1": DataFrame(история циклов), "engine_2": ... }
    engine_history = {}

    logging.info(f"🟢 CONSUMER: Запущен. Ожидание сырого потока {STREAM_NAME}...")
    
    while True:
        messages = r.xreadgroup(GROUP_NAME, "worker-1", {STREAM_NAME: '>'}, count=5, block=1000)
        
        if messages:
            for stream, msg_list in messages:
                for msg_id, msg_data in msg_list:
                    engine_id = msg_data['engine_id']
                    cycle = int(msg_data['cycle'])
                    
                    # 1. Извлекаем сырые данные сенсоров из сообщения
                    current_sensors = {s: float(msg_data[s]) for s in ACTIVE_SENSORS}
                    
                    # 2. Добавляем в историю двигателя
                    if engine_id not in engine_history:
                        engine_history[engine_id] = []
                    engine_history[engine_id].append(current_sensors)
                    
                    # Ограничиваем историю 15ю циклами (чтобы не переполнить память)
                    if len(engine_history[engine_id]) > 15:
                        engine_history[engine_id].pop(0)
                    
                    # 3. Если накопилось минимум 5 циклов — делаем предсказание
                    if len(engine_history[engine_id]) >= 5:
                        buffer_df = pd.DataFrame(engine_history[engine_id])
                        
                        # Считаем фичи на лету!
                        features = calculate_features(buffer_df)
                        df_pred = pd.DataFrame([features])
                        
                        # Инференс
                        proba = model.predict_proba(df_pred)[0][1]
                        prediction = 1 if proba >= 0.43 else 0
                        status = "🔴 ТО" if prediction == 1 else "🟢 НОРМА"
                        
                        # Выводим тренд вероятности
                        bar = "█" * int(proba * 20) + "░" * (20 - int(proba * 20))
                        logging.info(f"[{status}] Двигатель {engine_id:>3} | Цикл {cycle:>3} | Prob: {proba:.3f} |{bar}|")
                    
                    r.xack(STREAM_NAME, GROUP_NAME, msg_id)

if __name__ == "__main__":
    run_consumer()