import re

import pandas as pd

QUARTER_PATTERN = re.compile(r"\d{4}-Q\d")
HEADER_ROW = 6
LABEL_COLUMN = 1
SERIES_SOURCES = {
    "Прямі продажі": ("1 Type of sale", "Direct"),
    "Через посередників": ("1 Type of sale", "Intermediary"),
    "Перші покупці": ("2 Loan purpose", "First Time Buyer"),
    "Зміна житла": ("2 Loan purpose", "Home Movers (2nd Or Subsequent Buyers)"),
    "Рефінансування": ("2 Loan purpose", "Remortgagors"),
    "Викуп соцжитла": ("2 Loan purpose", "Council/Registered Social Landlord Tenant Exercising Their Right To Buy"),
    "Інше": ("2 Loan purpose", "Other"),
}


def read_sheet_series(path, sheet, label):
    table = pd.read_excel(path, sheet_name=sheet, header=None)
    header = table.iloc[HEADER_ROW]
    quarter_columns = [column for column in table.columns if isinstance(header[column], str)
                       and QUARTER_PATTERN.fullmatch(header[column])]
    row = table.index[table[LABEL_COLUMN].astype(str).str.strip() == label][0]
    index = pd.PeriodIndex([header[column] for column in quarter_columns], freq="Q")
    return pd.Series(table.loc[row, quarter_columns].astype(float).to_numpy(), index=index)


def load_mortgage_sales(path):
    return pd.DataFrame({name: read_sheet_series(path, sheet, label)
                         for name, (sheet, label) in SERIES_SOURCES.items()})
