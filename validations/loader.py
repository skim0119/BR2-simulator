import pandas as pd
import numpy as np


def load_data(csv_path: str, x_key: str, y_key: str, keys: list[str] | None = None):
    data = pd.read_csv(csv_path)

    # Convert to numpy
    activation = data[x_key]
    result = data[y_key]
    activation = np.array(activation)
    result = np.array(result)

    if keys is not None:
        info = {}
        for key in keys:
            info[key] = np.array(data[key])
        return activation, result, info
    return activation, result
