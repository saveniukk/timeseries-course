# Лабораторна робота №2

## Запуск
 
1. Python 3.12+.
2. Створити середовище: `python -m venv .venv`
3. Активувати його: `source .venv/bin/activate` (Linux, macOS) або `.venv\Scripts\activate` (Windows).
4. Встановити залежності: `pip install -r requirements.txt`
5. Запустити: `python main.py`
Програма виводить у консоль результати кроків 1–7 і верифікацію, а потім по черзі відкриває графіки. Наступний графік з'являється після закриття попереднього.
 
## Дані
 
Курс USD НБУ за 2025 рік лежить у `data/parsed_minfin_usd_2025.csv`. Щоб спарсити сайт заново, у `main.py` потрібно поставити `REFRESH_DATA = True`.
