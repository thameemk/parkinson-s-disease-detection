#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : feature_scaling.py
#  Author : thameem
#  Modified time : Thu, 24 Nov 2022 at 12:16 am India Standard Time
from sklearn.decomposition import PCA
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


def feature_scaling_pca(x_train, x_test):
    """
    unsupervised learning technique for reducing the dimensionality of data. It increases interpretability yet,
    at the same time, it minimizes information loss. It helps to find the most significant features in a dataset and
    makes the data easy for plotting in 2D and 3D

    """
    pca = PCA(n_components=2)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    variance = pca.explained_variance_ratio_

    return x_train, x_test, variance
