"""
Скрипт для быстрой проверки работоспособности приложения.
Запускать из терминала: 
```
    python test_app.py
```
"""


import time
import json
from pathlib import Path
from pprint import pprint
from typing import Union
from urllib.request import urlopen, Request


# Адрес сервера и путь к примерам
app_url = 'http://127.0.0.1:8000'
examples_file = Path('data', 'examples.json')
print('kjkjkj', examples_file)

def send_request(endpoint: str, data: Union[bytes, None] = None) ->None:
    """Отправляет запрос на сервер и печатает ответ."""

    start_time = time.time()
    if data is not None:
        request = Request(app_url + endpoint, data=data, method='POST', 
                          headers={"Content-Type": "application/json"})
    else:
        request = Request(app_url + endpoint)
    with urlopen(request) as response:
        result = json.loads(response.read())
    pprint(result)
    print(f'Время запроса {time.time() - start_time} секунд')
    print('-' * 80)


def test_app() -> None:
    """Отправляет все возможные запросы на сервер и печатает ответ."""

    # Загрузка примеров
    with open(examples_file, 'rb') as file:
        examples = json.load(file)
        # print('iiiiiiiiiiiiiiiiiiiiii', examples[0])

    print('-' * 80)
    print('Тесты запросов к приложению.')
    print('-' * 80)

    print('Запрос статуса.')
    send_request('/status')

    print('Запрос метаданных.')
    send_request('/version')

    print('Запрос предсказания класса для одного объекта.')
    data = json.dumps(examples[0]).encode("utf-8")
    print('88888888888888888888888888888888888888888', data)
    send_request('/predict', data)

    # print('Запрос предсказания вероятности класса для одного объекта.')
    # data = json.dumps(examples[1]).encode("utf-8")
    # send_request('/predict_proba', data)
    #
    # print('Запрос предсказания класса для множества объектов.')
    # data = json.dumps(examples).encode("utf-8")
    # send_request('/predict_many', data)
    #
    # print('Запрос предсказания вероятности класса для множества объектов.')
    # data = json.dumps(examples).encode("utf-8")
    # send_request('/predict_proba_many', data)


if __name__ == '__main__':
    test_app()
    try: 
        test_app()
    except:
        print('Непредвиденная ошибка. Проверьте, включен ли сервер с приложением.')
