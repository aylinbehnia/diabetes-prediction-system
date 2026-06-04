from sklearn.model_selection import GridSearchCV

def tune_knn(model, X, y):
    params = {"n_neighbors":[3,5,7,9]}
    grid = GridSearchCV(model, params, cv=5)
    grid.fit(X, y)
    return grid.best_params_