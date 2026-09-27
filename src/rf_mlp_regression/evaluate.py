"""Model evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def evaluate_predictions(y_true, y_pred, y_train, label: str = "") -> float:
    """Compute RMSE and relative error against the target's range in train."""
    rmse = root_mean_squared_error(y_true, y_pred)
    rel_error = rmse / (y_train.max() - y_train.min()) * 100

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}RMSE: {rmse:.3f}")
    print(f"{prefix}Relative error: {rel_error:.2f}%")

    return rmse


def top_errors(y_true, preds, n: int = 5) -> pd.DataFrame:
    """Return the n predictions with the largest relative percentage error."""
    err = pd.DataFrame({"Actual": np.asarray(y_true), "Predicted": np.asarray(preds).ravel()})
    err["Relative_Error_%"] = np.abs(err["Actual"] - err["Predicted"]) / err["Actual"] * 100
    return err.nlargest(n, "Relative_Error_%").round(2).reset_index(drop=True)
