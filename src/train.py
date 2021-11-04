"""
This script trains a linear regressor with scikit-learn and saves it as .joblib file.
"""
if __name__ == '__main__':
    from pathlib import Path
    from joblib import dump

    from sklearn import datasets, linear_model

    X, y = datasets.load_iris(return_X_y=True)
    regressor = linear_model.LinearRegression()
    regressor.fit(X, y)
    dump(regressor, Path(__file__).parent / 'resources' / 'regressor.joblib')
