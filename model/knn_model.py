from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier()),
        ]
    )


def parameter_grid():
    return {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "classifier__weights": ["uniform", "distance"],
    }
