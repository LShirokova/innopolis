import os
import time
import joblib
import logging
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive Maintenance Real-Time API")

# Подключение к Online Feature Store
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'), 
    port=os.getenv('REDIS_PORT', 6379), 
    decode_responses=True
)

# Загрузка модели при старте
model = joblib.load("models/best_model.pkl")
THRESHOLD = 0.43

class PredictionResponse(BaseModel):
    engine_id: str
    prediction: int
    failure_probability: float
    inference_time_ms: float
    recommendation: str

@app.get("/health")
def health():
    return {"status": "ok", "redis_connected": redis_client.ping()}

@app.post("/predict", response_model=PredictionResponse)
def predict_engine_failure(engine_id: str):
    """Получает предсказание для двигателя из Online Feature Store."""
    start_time = time.perf_counter()
    
    # 1. Достаем фичи из Redis (Online Store) - это занимает <1ms
    feature_key = f"features:{engine_id}"
    features = redis_client.hgetall(feature_key)
    
    if not features:
        raise HTTPException(status_code=404, detail=f"Фичи для двигателя {engine_id} не найдены в Online Store")
    
    # 2. Подготовка данных для модели (сортируем колонки как при обучении)
    import pandas as pd
    df = pd.DataFrame([features])
    df = df.astype(float)
    
    # ВАЖНО: Убедимся, что порядок колонок совпадает с моделью
    # (В реальном проекте список фичей сохраняется отдельно при тренировке)
    
    # 3. Инференс
    proba = model.predict_proba(df)[0][1]
    prediction = 1 if proba >= THRESHOLD else 0
    
    inference_time_ms = (time.perf_counter() - start_time) * 1000
    
    return PredictionResponse(
        engine_id=engine_id,
        prediction=prediction,
        failure_probability=round(proba, 4),
        inference_time_ms=round(inference_time_ms, 2),
        recommendation="ТРЕБУЕТСЯ ОБСЛУЖИВАНИЕ" if prediction == 1 else "Нормальная работа"
    )