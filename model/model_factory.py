from model.decision_tree_model import create_model as create_decision_tree
from model.decision_tree_model import parameter_grid as decision_tree_grid
from model.knn_model import create_model as create_knn
from model.knn_model import parameter_grid as knn_grid
from model.logistic_regression_model import create_model as create_logistic_regression
from model.logistic_regression_model import parameter_grid as logistic_regression_grid
from model.naive_bayes_model import create_model as create_naive_bayes
from model.naive_bayes_model import parameter_grid as naive_bayes_grid
from model.random_forest_model import create_model as create_random_forest
from model.random_forest_model import parameter_grid as random_forest_grid


def build_search_specs():
    """Return the five assignment classifiers and their training-only CV grids."""
    return {
        "Logistic Regression": (create_logistic_regression(), logistic_regression_grid()),
        "Decision Tree": (create_decision_tree(), decision_tree_grid()),
        "K-Nearest Neighbor": (create_knn(), knn_grid()),
        "Naive Bayes": (create_naive_bayes(), naive_bayes_grid()),
        "Random Forest": (create_random_forest(), random_forest_grid()),
    }
