from enum import Enum

class ModelType(Enum):
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ENSEMBLE = "ensemble"  # Combination of tree-based models