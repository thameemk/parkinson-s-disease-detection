#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : predictors.py
#  Author : thameem
#  Modified time : Thu, 24 Nov 2022 at 10:08 pm India Standard Time
import enum


class Predictors(enum.Enum):
    XG_BOOST = "XG_BOOST"
    KNN = "KNN"
    SVM = "SVM"
    RANDOM_FOREST = "RANDOM_FOREST"
