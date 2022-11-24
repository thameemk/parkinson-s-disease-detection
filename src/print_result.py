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
    print("====", classifier, "====\n\n\nConfusion Matrix:\n\n", confusion_matrix, "\n\n Accuracy: ", accuracy * 100,
          "\n\n===================\n\n")
