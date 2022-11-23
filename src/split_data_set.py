#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : split_data_set.py
#  Author : thameem
#  Modified time : Sat, 19 Nov 2022 at 12:08 am India Standard Time
from sklearn.model_selection import train_test_split


def split_train_and_test_data(x, y):
    """
    split the given data set into training & testing datas

    """

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test
