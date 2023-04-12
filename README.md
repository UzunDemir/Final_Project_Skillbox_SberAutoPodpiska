# Предсказание целевых действий на сайте 'СберАвтоподписка'

Проект в качестве итогового задания по курсу 'Введение в Data Science' от Skillbox.

Выполнил: Демир Узун

**Цель** - разработать приложение, которое будет предсказывать совершение пользователем одного из целевых действий - 'Оставить заявку' или 'Заказать звонок' по времени визита 'visit_\*', рекламным меткам 'utm_\*', характеристикам устройства 'device_\*' и местоположению 'geo_\*'. Это должен быть сервис, который будет брать на
вход все атрибуты, типа utm_*, device_*, geo_*, и отдавать на выход 0/1 (1 — если пользователь совершит любое целевое действие).

Целевые **метрики**: `roc-auc` > 0.65, время предсказания не более 3 секунд.

Полностью задание можно прочитать [отсюда](https://github.com/UzunDemir/Final_Project_Skillbox_SberAutoPodpiska/blob/main/description.md) или скачать в виде [файла-pdf](https://drive.google.com/file/d/1R-Lk45ZeXPf6v13_MfV-8qYp_1wv0N2S/view).

Данные для выполнения практического задания можно скачать [отсюда](https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw).






## Блокноты

Ноутбуки с разведовательным анализом данных и исследованием моделей для задачи:

* [step_1_data_analises.ipynb](https://github.com/UzunDemir/Final_Project_Skillbox_SberAutoPodpiska/blob/main/step_1_data_analises.ipynb). Здесь я провожу анализ данных клиентских сессий сервиса 'СберАвтоподписка'. В этом ноутбуке мы реализуем три стадии машинного обучения:
* Business understanding (выстроим понимание бизнесс-процессов сервиса 'СберАвтоподписка' и возможности для его улучшения на основе данных клиентских сессий);
* Data understanding (изучим предоставленные данные и выявим их связь с реальными процессами)
* Data preparation (подготовим данные для проведения обучения моделей)

вынесены в отдельную папку `notebooks`. К ним же скопирован файл с дополнительными данными `additional_data.py` для независимости от остального проекта.

## Обучение модели

Для создания модели необходимо запустить скрипт `create_model.py` (файл `additional_data.py` и папки с данными `data` и моделями `models` должны находится в той же директории).

Или можно создать модель, запустив ноутбук с исследованиями моделей.

## Запуск приложения

Модель оформлена в качестве отдельного приложения, находящегося в папке `app`. Чтобы его запустить, необходимо установить библиотеки из `requirements.txt` и в корневой папке проекта выполнить команду:  

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Также приложение может быть запущено с помощью Docker. Для этого нужно выполнить в корневой папке проекта команды для построения образа и запуска `docker-compose` (возможно с `sudo`):
```
docker-compose build
docker-compose up

# для выключения docker-compose
docker-compose down
```
Папка с моделями `models` подключается как внешний том в `docker-compose.yml`.

В любом случае приложение будет доступно по адресу `http://127.0.0.1:8000`.

## Методы API

Для работы с приложением можно использовать запросы: 
+ `/status` (get) - для получения статуса сервиса;
+ `/version` (get) - для получения версии и метаданных модели;
+ `/predict` (post) - для предсказания класса одного объекта;
+ `/predict_many` (post) - для предсказания класса множества объектов;
+ `/predict_proba` (post) - для предсказания вероятности положительного класса одного объекта;
+ `/predict_proba_many` (post) - для предсказания вероятности положительного класса множества объектов.

Все эти методы можно быстро протестировать с помощью скрипта `test_app.py`.
