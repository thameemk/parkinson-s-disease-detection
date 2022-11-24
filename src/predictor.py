#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : predictor.py
#  Author : thameem
#  Modified time : Thu, 24 Nov 2022 at 10:07 pm India Standard Time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.enums import Predictors


def predictor(x_train, y_train, x_test, preditor: Predictors):
    """
    fitting the data to respective classifier and predict the result
    Args:
        x_train:
        y_train:
        x_test:
        preditor:

    Returns:

    """

    if preditor == Predictors.KNN:
        classifier = KNeighborsClassifier(n_neighbors=8, p=2, metric='minkowski')
    elif preditor == Predictors.SVM:
        classifier = SVC()
    elif preditor == Predictors.XG_BOOST:
        classifier = XGBClassifier()
    elif preditor == Predictors.RANDOM_FOREST:
        classifier = RandomForestClassifier(n_estimators=16, criterion="entropy", random_state=0)
    else:
        raise NotImplemented()

    classifier.fit(x_train, y_train)

    return classifier.predict(x_test)
