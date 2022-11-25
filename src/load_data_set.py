#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : load_data_set.py
#  Author : thameem
#  Modified time : Fri, 18 Nov 2022 at 11:30 pm India Standard Time
import pandas as pd
from numpy import ndarray


def load_data_set() -> [ndarray, ndarray]:
    """
    helps to load the data set

    Returns: data set
    """
    dataset = pd.read_csv('../data/parkinsons.data')

    print(f"\nReading data set.......\n{dataset.head()}")

    x = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23]].values
    y = dataset.iloc[:, 17].values

    return x, y
