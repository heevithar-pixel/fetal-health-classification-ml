from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE


def create_model():
    return DecisionTreeClassifier(random_state=RANDOM_STATE)


def parameter_grid():
    return {
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 5, 10],
    }
