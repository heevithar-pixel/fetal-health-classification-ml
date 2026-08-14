from sklearn.naive_bayes import GaussianNB


def create_model():
    return GaussianNB()


def parameter_grid():
    return {
        "var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
    }
