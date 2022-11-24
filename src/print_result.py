#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : print_result.py
#  Author : thameem
#  Modified time : Thu, 24 Nov 2022 at 11:19 pm India Standard Time
from src.enums import Classifiers


def print_result(classifier: Classifiers, confusion_matrix, accuracy) -> None:
    """
    print the results
    Args:
        classifier: which  classifier is used
        confusion_matrix: the confusion matrix
        accuracy: accuracy of the model used
    """

    print(f"\n===== {classifier.value} =====\nConfusion Matrix: \n {confusion_matrix}\nAccuracy: {accuracy * 100}")
