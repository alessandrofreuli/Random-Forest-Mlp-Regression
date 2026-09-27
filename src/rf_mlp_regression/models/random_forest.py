"""Random Forest model with GridSearchCV."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from .. import config


def train_random_forest(X_train_prep, y_train) -> RandomForestRegressor:
    """Train a Random Forest with hyperparameter search via grid search."""
    rf = RandomForestRegressor(random_state=config.RANDOM_STATE)

    grid_search = GridSearchCV(
        rf,
        config.RF_PARAM_GRID,
        cv=config.RF_CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train_prep, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.3f}")

    return grid_search.best_estimator_
