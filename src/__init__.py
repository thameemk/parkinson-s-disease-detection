#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : __init__.py
#  Author : thameem
#  Modified time : Fri, 18 Nov 2022 at 11:04 pm India Standard Time

from .load_data_set import load_data_set
from .split_data_set import split_train_and_test_data
from .feature_scaling import feature_scaling_standard_scalar, feature_scaling_pca
from .get_cm_and_accuracy import get_cm_and_accuracy
from .predictor import predictor
from .print_result import print_result
