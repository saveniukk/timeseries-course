# timeseries-course

## Запуск ЛР №1

```
cd lab-1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

На Windows активація: `.venv\Scripts\activate`.

Дані вже збережені в `data/parsed_minfin_usd_2025.csv`. Щоб спарсити сайт заново, поставити `REFRESH_DATA = True` у `main.py`.
