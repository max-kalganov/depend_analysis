import pandas as pd


def read_data():
    path = "data/Cars.xls"
    return pd.read_excel(path)


if __name__ == '__main__':
    print(read_data())
