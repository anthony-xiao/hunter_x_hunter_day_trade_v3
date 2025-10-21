from enum import Enum

class ModelType(Enum):
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"  # Combination of above three