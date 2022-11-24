#  Project : Detection of Parkinson's Disease Using Vocal Features: An Eigen Approach
#  Filename : predictors.py
#  Author : thameem
#  Modified time : Thu, 24 Nov 2022 at 10:08 pm India Standard Time
import enum


class Classifiers(enum.Enum):
    XG_BOOST = "XGBoost"
    KNN = "K-Nearest Neighbor"
    SVM = "Support Vector Machine"
    RANDOM_FOREST = "Random Forest"
