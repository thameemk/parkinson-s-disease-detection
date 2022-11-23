#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : feature_scaling.py
#  Author : thameem
#  Modified time : Thu, 24 Nov 2022 at 12:16 am India Standard Time
from sklearn.preprocessing import StandardScaler


def feature_scaling_standard_scalar(x_train, x_test):
    """
    standardizes a feature by subtracting the mean and then scaling to unit variance

    Args:
        x_train:
        x_test:

    Returns:
    """
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    return x_train, x_test
