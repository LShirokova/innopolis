import time
import redis
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

STREAM_NAME = "raw_sensor_stream"

def run_producer():
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
    
    # ВАЖНО: Берем тестовый датасет (двигатели работают, пока не сломаются)
    test_file = 'data/raw/test_FD001.txt'
    cols = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    print(f"🚀 PRODUCER: Стриминг сырых данных из {test_file}...")
    df = pd.read_csv(test_file, sep='\s+', header=None, names=cols)
    
    # Удаляем колонки с настройками оператора, оставляем только цикл и сенсоры
    sensor_cols = [c for c in df.columns if 'sensor_' in c]
    
    count = 0
    for _, row in df.iterrows():
        engine_id = str(int(row['engine_id']))
        cycle = str(int(row['cycle']))
        
        # Формируем словарь с сырыми данными
        message = {"engine_id": engine_id, "cycle": cycle}
        for s in sensor_cols:
            message[s] = str(float(row[s]))
            
        r.xadd(STREAM_NAME, message)
        count += 1
        
        # Стримим со скоростью 10 сообщений в секунду
        time.sleep(0.1)
        
        if count % 50 == 0:
            print(f"📤 Отправлено {count} сырых сообщений...")
            
    print(f"✅ PRODUCER: Датасет завершен.")

if __name__ == "__main__":
    run_producer()