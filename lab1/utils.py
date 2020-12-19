from collections import defaultdict

import pandas as pd


def read_data() -> pd.DataFrame:
    path = "data/regres.txt"
    matrix = defaultdict(list)
    with open(path, "r") as input_data:
        for line in input_data.readlines():
            x1, x2, x3, x4 = line.split()
            matrix["air_speed"].append(float(x1))
            matrix["temperature"].append(float(x2))
            matrix["acid_conc"].append(float(x3))
            matrix["ammonia_losses"].append(float(x4))

    return pd.DataFrame(matrix)


if __name__ == '__main__':
    data = read_data()
    print(data)
