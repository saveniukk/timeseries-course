import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

SITE_ROOT = "https://index.minfin.com.ua"
YEAR_URL = SITE_ROOT + "/ua/exchange/archive/nbu/curr/2025/"
DAY_LINK_PATTERN = re.compile(r"/ua/exchange/archive/nbu/curr/(\d{4}-\d{2}-\d{2})/$")
CURRENCY_CODE = "USD"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) timeseries-lab"}
REQUEST_TIMEOUT_SECONDS = 30
PAUSE_BETWEEN_REQUESTS_SECONDS = 0.3
MAX_ATTEMPTS = 3
RATE_COLUMN = 4
CODE_COLUMN = 1


def fetch_html(session, url):
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = session.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            time.sleep(PAUSE_BETWEEN_REQUESTS_SECONDS * (attempt + 1))
    raise ConnectionError(f"Не вдалося завантажити {url}")


def collect_day_links(year_html):
    soup = BeautifulSoup(year_html, "lxml")
    links = {}
    for anchor in soup.find_all("a", href=True):
        match = DAY_LINK_PATTERN.search(anchor["href"])
        if match:
            links[match.group(1)] = SITE_ROOT + anchor["href"]
    return dict(sorted(links.items()))


def parse_usd_rate(day_html):
    soup = BeautifulSoup(day_html, "lxml")
    for row in soup.select("table.zebra tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) > RATE_COLUMN and cells[CODE_COLUMN] == CURRENCY_CODE:
            return float(cells[RATE_COLUMN].replace("\xa0", "").replace(" ", "").replace(",", "."))
    raise ValueError("Рядок з курсом USD не знайдено")


def parse_year_rates():
    session = requests.Session()
    day_links = collect_day_links(fetch_html(session, YEAR_URL))
    records = []
    for index, (date, url) in enumerate(day_links.items(), start=1):
        try:
            records.append((date, parse_usd_rate(fetch_html(session, url))))
        except ValueError as error:
            print(f"День {date} пропущено: {error}")
        if index % 50 == 0:
            print(f"Оброблено сторінок днів: {index} з {len(day_links)}")
        time.sleep(PAUSE_BETWEEN_REQUESTS_SECONDS)
    frame = pd.DataFrame(records, columns=["date", "usd_rate"])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    return clean_series(frame)


def clean_series(frame):
    frame = frame.dropna().drop_duplicates(subset="date")
    return frame.sort_values("date").reset_index(drop=True)


def load_series(csv_path, refresh):
    if csv_path.exists() and not refresh:
        print(f"Використано збережений файл: {csv_path.name}")
        return read_csv(csv_path)
    try:
        frame = parse_year_rates()
    except (ConnectionError, ValueError) as error:
        if not csv_path.exists():
            raise
        print(f"Парсинг не вдався ({error}); використано збережений файл: {csv_path.name}")
        return read_csv(csv_path)
    frame.to_csv(csv_path, index=False)
    print(f"Парсинг завершено, збережено {len(frame)} рядків у {csv_path.name}")
    return frame


def read_csv(csv_path):
    return pd.read_csv(csv_path, parse_dates=["date"])
