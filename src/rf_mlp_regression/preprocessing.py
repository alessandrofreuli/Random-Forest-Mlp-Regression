"""Preprocessing pipeline for numeric and categorical features."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(X_train) -> ColumnTransformer:
    """Build the ColumnTransformer for numeric + categorical features."""
    num_cols = X_train.select_dtypes(include=[np.number, bool]).columns
    cat_cols = X_train.select_dtypes(include=["object", "str"]).columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(sparse_output=False)),
    ])

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])
    return preprocessing


def fit_transform_data(preprocessing: ColumnTransformer, X_train, X_test):
    """Fit on train and transform both train/test."""
    X_train_prep = preprocessing.fit_transform(X_train)
    X_test_prep = preprocessing.transform(X_test)
    return X_train_prep, X_test_prep
