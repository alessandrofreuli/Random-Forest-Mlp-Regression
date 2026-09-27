"""Dataset loading and splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_data(path=None) -> pd.DataFrame:
    """Load the raw dataset and drop unnecessary columns."""
    path = path or config.DATA_PATH
    data = pd.read_csv(path)
    data = data.drop(columns=config.DROP_COLUMNS, errors="ignore")
    return data


def train_test_split_data(data: pd.DataFrame):
    """Split into train/test sets and separate features from the target."""
    train_set, test_set = train_test_split(
        data, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    X_train = train_set.drop(config.TARGET, axis=1)
    y_train = train_set[config.TARGET].copy()

    X_test = test_set.drop(config.TARGET, axis=1)
    y_test = test_set[config.TARGET].copy()

    return X_train, X_test, y_train, y_test
