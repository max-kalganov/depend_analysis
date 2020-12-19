import pandas as pd


def read_data():
    path = "data/Barotrop.xls"
    return pd.read_excel(path)
