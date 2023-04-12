
"""
Скрипт для быстрой проверки работоспособности приложения.
Скрипт считывает данные из файла с 5 примерами и поочередно
подает их на самую новую обученную модель-пайплайн для определения предикта.
Скрипт выводит название и расположение модели, session_id,
предсказание и время выполнения запроса.

Для получения вероятности предсказания: prediction = model.predict_proba(example)[:, 1]
Для рлдучения класса предсказания prediction = model.predict(example)
Запускать из терминала:
```
    python predict_test.py
```
"""
import warnings
import dill
import json
import pandas as pd
import additional_data as add
from datetime import datetime
import time

from os import PathLike
from pathlib import Path
from sklearn.base import BaseEstimator
warnings.filterwarnings('ignore')   # никогда не печатать соответствующие предупреждения
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
