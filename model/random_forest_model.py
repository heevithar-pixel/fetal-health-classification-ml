from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE


def create_model():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def parameter_grid():
    return {
        "max_depth": [None, 10, 16],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }
