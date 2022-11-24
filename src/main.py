#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : main.py
#  Author : thameem
#  Modified time : Thu, 24 Nov 2022 at 10:52 pm India Standard Time
from src import load_data_set, split_train_and_test_data, feature_scaling_standard_scalar, predictor, \
    get_cm_and_accuracy, print_result
from src.enums import Classifiers

if __name__ == '__main__':
    x, y = load_data_set()

    x_train, x_test, y_train, y_test = split_train_and_test_data(x, y)

    x_train, x_test = feature_scaling_standard_scalar(x_train, x_test)

    for classifier in Classifiers:
        confusion_matrix, accuracy = get_cm_and_accuracy(y_test, predictor(x_train, y_train, x_test, classifier))

        print_result(classifier, confusion_matrix, accuracy)
