from enum import Enum

class ModelType(Enum):
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    ENSEMBLE = "ensemble"  # Combination of above three