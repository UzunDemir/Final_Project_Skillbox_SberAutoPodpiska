import dill
import pandas as pd
import json

from os import PathLike
from pathlib import Path
from sklearn.base import BaseEstimator

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


import additional_data as add
from datetime import datetime
import time

from os import PathLike
from pathlib import Path
from sklearn.base import BaseEstimator

# Шаблон названия моделей
model_name_pattern = 'model_*.pkl'


# Функция таймера для замера времени выполнения операций
def time_it():
    elapsed_time = datetime.now() - start_time
    return elapsed_time


start_time = datetime.now()


# Необходимая функция для подготовки новых признаков
def _distance_category(distance: float) -> str:
    """Возвращает категорию расстояния до Москвы."""

    if distance == -1:
        return 'no distance'
    elif distance == 0:
        return 'moscow'
    elif distance < 100:
        return '< 100 km'
    elif distance < 500:
        return '100-500 km'
    elif distance < 1000:
        return '500-1000 km'
    elif distance < 3000:
        return '1000-3000 km'
    else:
        return '>= 3000 km'


# Функция загрузки модели
def load_model(folder: PathLike) -> BaseEstimator:
    """Загружает последнюю модель из заданной папки."""

    # Получим список моделей в папке
    folder = Path(folder)
    model_files = list(folder.glob(model_name_pattern))

    # Если модели есть, загрузим последнюю
    if model_files:
        last_model = sorted(model_files)[-1]
        print('Используем самую новую обученную модель:', last_model)
        with open(last_model, 'rb') as file:
            model = dill.load(file)
        return model

    # Иначе вызовем ошибку
    else:
        raise FileNotFoundError('Нет моделей в указанной папке.')


# Загрузка модели

model = load_model('models')


for key, value in model.metadata.items():
    print(key, ":", value)

# Загрузка примеров из json-файла для проверки работы модели
with open('data/examples.json', 'rb') as file:
    examples = json.load(file)
    df = pd.DataFrame.from_dict(examples)

for i in range(len(examples)):
    print('=' * 100)
    example = df.iloc[[i]]
    #prediction = model.predict_proba(example)[:, 1]
    prediction = model.predict(example)
    print(example.session_id, prediction, time_it())

class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    event_value: float


@app.get('/status')
def status():
    return "I am OK!!!!!!!!!!!!!!!!!!!!!!!!"


@app.get('/version')
def version():
    return model.metadata


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'event_value': '2222222222222'#y[0]
    }

#uvicorn api:app --reload