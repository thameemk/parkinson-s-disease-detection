#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : get_cm_and_accuracy.py
#  Author : thameem
#  Modified time : Thu, 24 Nov 2022 at 12:32 am India Standard Time
from sklearn.metrics import confusion_matrix, accuracy_score


def get_cm_and_accuracy(y_test, y_pred):
    """
    calculate the confusion matrix and accuracy
    Args:
        y_test:
        y_pred:

    Returns:

    """
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return cm, accuracy
