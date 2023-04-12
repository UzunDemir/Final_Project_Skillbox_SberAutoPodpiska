# Разработка и обучение модели, предсказывающей совершение клиентами целевых действий на сайте 'СберАвтоподписка'

Проект в качестве итогового задания по курсу 'Введение в Data Science' от Skillbox.

Выполнил: Демир Узун


![Аренда авто по подписке – СберАвтоподписка _ Сервис подписки на автомобили от 6 месяцев до 3 лет - Google Chrome 2023-04-12 16-08-35_Trim (1)](https://user-images.githubusercontent.com/94790150/231468350-3204455c-723c-40eb-8ad6-be7d48e374ad.gif)

**Цель** - разработать приложение, которое будет предсказывать совершение пользователем одного из целевых действий - 'Оставить заявку' или 'Заказать звонок' по времени визита 'visit_\*', рекламным меткам 'utm_\*', характеристикам устройства 'device_\*' и местоположению 'geo_\*'. Это должен быть сервис, который будет брать на
вход все атрибуты, типа utm_*, device_*, geo_*, и отдавать на выход 0/1 (1 — если пользователь совершит любое целевое действие).

Целевые **метрики**: `roc-auc` > 0.65, время предсказания не более 3 секунд.

Полностью задание можно прочитать [отсюда](https://github.com/UzunDemir/Final_Project_Skillbox_SberAutoPodpiska/blob/main/description.md) или скачать в виде [файла-pdf](https://drive.google.com/file/d/1R-Lk45ZeXPf6v13_MfV-8qYp_1wv0N2S/view).

Данные для выполнения практического задания можно скачать [отсюда](https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw).

## О компании [«СберАвтоподписка»](https://sberautopodpiska.ru/)

«СберАвтоподписка» — это сервис долгосрочной аренды автомобилей для физлиц.

Клиент платит фиксированный ежемесячный платёж и получает в пользование машину на срок от шести месяцев до трёх лет. 

Также в платёж включены:
* страхование (КАСКО, ОСАГО, ДСАГО);
* техническое обслуживание и ремонт;
* сезонная смена шин и их хранение;
* круглосуточная служба поддержки.

За дополнительную сумму можно приобрести услугу консьерж-сервиса — доставку автомобиля до сервисного центра и обратно на техническое обслуживание, сезонную замену шин, ремонт.

## Описание структуры проекта:
* `.idea`, `.ipynb_checkpoints`, `__pycache__` - служебные папки, сгенерированные Pycharm
* `data`- папка, где хранятся данные для проверки работоспособности приложения. `examples.json` - содержит в себе список из 5 примеров для проверки.
* `models` - папка, куда записываются готовые модели.
* `additional_data.py`- дополнительные данные для генерации признаков.
* `api.py` - скрипт для проверки API-запросов.
* `create_model.py`- скрипт, который создает пайплайн и обучает модель.
* `description.md`- описание целевого задания.
* `other.ipynb` - не используется
* `predict_test.py`- скрипт, который проверяет работу приложения
* `presentation.pdf` - презентация проекта
* `requirements.txt`- необходимые библиотеки для установки. Pycharm автоматически устанавливает их из списка в этом файле.
* `step_1_data_analises.ipynb` - ноутбук анализ датасетов.
* `step_2_model_research.ipynb`- ноутбук исследования и моделирование.



## Последовательность работы

* Скопируйте данный репозиторий к себе на компьютер.
* Скачайте [данные](https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw) GA Sessions (ga_sessions.csv) и GA Hits (ga_hits.csv) и разместите их в папке `data`, которую нужно создать на уровень выше вашего проекта. 
* После этого можно запустить ноутбуки [step_1_data_analises.ipynb](https://github.com/UzunDemir/Final_Project_Skillbox_SberAutoPodpiska/blob/main/step_1_data_analises.ipynb) и [step_2_model_research.ipynb](https://github.com/UzunDemir/Final_Project_Skillbox_SberAutoPodpiska/blob/main/step_2_model_research.ipynb) и пошагово отработать каждую ячейку для анализа происходящего исследования.

## Анализ данных клиентских сессий сервиса 'СберАвтоподписка'

Всего целевых действий = 104908 из 15726470 что составляет  0.67 %

sub_car_claim_click = 37928 === 0.24 %

sub_open_dialog_click = 25870 === 0.16 %

sub_submit_success = 18439 === 0.12 %

sub_car_claim_submit_click = 12359 === 0.08 %

sub_call_number_click = 3653 === 0.02 %

sub_callback_submit_click = 3074 === 0.02 %

sub_car_request_submit_click = 2966 === 0.02 %

sub_custom_question_submit_click = 619 === 0.0 %

![image](https://user-images.githubusercontent.com/94790150/231448673-8118d2ee-0bf3-4e26-b2a7-4031f043d668.png)

![image](https://user-images.githubusercontent.com/94790150/231449274-3b105023-fe5b-4604-a9d7-a4582a603f32.png)

Распределение целевой переменной:

* False    90.43%
* NaN       6.87%
* True      2.70%

![image](https://user-images.githubusercontent.com/94790150/231449811-0c9cbe6f-0fbf-41fd-8e5f-1b23d8fe028f.png)

![image](https://user-images.githubusercontent.com/94790150/231449936-e736d330-0cbe-4608-8875-3da1634b405a.png)

![image](https://user-images.githubusercontent.com/94790150/231450324-565b26a6-7967-4464-9ff7-f999659b5feb.png)

![image](https://user-images.githubusercontent.com/94790150/231450483-2da683a0-64be-4a1a-885c-fd1e713d8aae.png)

![image](https://user-images.githubusercontent.com/94790150/231450729-4341c0b5-7c76-40f4-b61e-aee0c40fbbf9.png)





* Из программы Pycharm или командной строки можно запустить обучение модели с помощью скрипта `create_model.py` (файл `additional_data.py` и папки с данными `data` и моделями `models` должны находится в той же директории). В файле `additional_data.py` собраны дополнительные данные, которые нужны для генерации признаков. 







## Блокноты

Ноутбуки с разведовательным анализом данных и исследованием моделей для задачи:

* [step_1_data_analises.ipynb](https://github.com/UzunDemir/Final_Project_Skillbox_SberAutoPodpiska/blob/main/step_1_data_analises.ipynb). Здесь мы проводим анализ данных клиентских сессий сервиса 'СберАвтоподписка'. В этом ноутбуке реализованы три стадии машинного обучения: Business understanding, Data understanding, Data preparation.
* [step_2_model_research.ipynb](https://github.com/UzunDemir/Final_Project_Skillbox_SberAutoPodpiska/blob/main/step_2_model_research.ipynb). Здесь мы непосредственно будем заниматься исследованиями, разработкой и оптимизацией модели.


## Обучение модели

Для создания модели необходимо запустить скрипт `create_model.py` (файл `additional_data.py` и папки с данными `data` и моделями `models` должны находится в той же директории).

Процесс создания пайплайна полностью информативен. В каждый момент времени можно знать какая проходит процедура и сколко времени ушло на выполнение этой процедуры.
Кстати, полностью весь процесс создания, обучения и оптимизации модели занимает всего около 10 минут.

![Final_Project_Skillbox_SberAutoPodpiska – create_model py 2023-04-12 09-36-22](https://user-images.githubusercontent.com/94790150/231450940-3b3cd9b6-e70f-4a8c-bc19-0a1e7ed9c6f2.gif)

Или можно создать модель, запустив ноутбук с исследованиями моделей.

## Запуск приложения

Модель можно запустить стационарно в виде скрипта `predict_test.py` или в качестве API-приложения. Чтобы его запустить, необходимо установить библиотеки из `requirements.txt` и в корневой папке проекта выполнить команду:  

Время предсказания приложения в десктопной версии составляет всего около 5 милисекунд.

![Final_Project_Skillbox_SberAutoPodpiska – predict_test py 2023-04-12 15-09-46 (1)](https://user-images.githubusercontent.com/94790150/231453881-5589eb0d-be34-4774-8313-5c499eec7b06.gif)

```
uvicorn api:app --reload
```
Запуск APi-приложения также очень быстрый.

![Final_Project_Skillbox_SberAutoPodpiska – predict_test py 2023-04-12 15-16-19_Trim (1)](https://user-images.githubusercontent.com/94790150/231461464-3f683bc2-5cb6-4219-b13e-94f415c4e75e.gif)

Время предсказания приложения в API версии также составляет всего около 5 милисекунд.

![http___127 0 0 1_8000_predict - My Workspace и еще 1 страница — Личный_ Microsoft_ Edge 2023-04-12 15-51-22 (1)](https://user-images.githubusercontent.com/94790150/231463498-a02d5d0f-a96d-48e4-876c-0b0408fa9531.gif)

В этом случае приложение будет доступно по адресу `http://127.0.0.1:8000`.

## Методы API

![http___127 0 0 1_8000_predict - My Workspace и еще 1 страница — Личный_ Microsoft_ Edge 2023-04-12 16-01-49 (1)](https://user-images.githubusercontent.com/94790150/231466542-5d00bb1a-f85b-440e-8c55-f2c73208a395.gif)


Для работы с приложением можно использовать запросы: 
+ `/status` (get) - для получения статуса сервиса;

![image](https://user-images.githubusercontent.com/94790150/231472065-675a304f-ebae-4bcd-8823-747aaa70d506.png)

+ `/version` (get) - для получения версии и метаданных модели;

![image](https://user-images.githubusercontent.com/94790150/231472315-f8cb48cc-88bc-4968-9903-1f673e298a16.png)


+ `/predict` (post) - для предсказания класса одного объекта;

![image](https://user-images.githubusercontent.com/94790150/231471817-18c57fd0-f8cd-4227-a705-bdba71f7c890.png)



