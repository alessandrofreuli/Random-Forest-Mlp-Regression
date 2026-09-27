"""Entry point: runs the full analysis and modeling pipeline."""
from rf_mlp_regression import config
from rf_mlp_regression.data import load_data, train_test_split_data
from rf_mlp_regression.preprocessing import build_preprocessing_pipeline, fit_transform_data
from rf_mlp_regression.models import train_random_forest, train_mlp, predict_mlp
from rf_mlp_regression.evaluate import evaluate_predictions, top_errors
from rf_mlp_regression.visualize import plot_model_comparison


def _section(title: str):
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def main():
    # 1. Load data
    _section("Dataset")
    data = load_data()
    print(f"Shape: {data.shape}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(data)

    # 3. Preprocessing
    preprocessing = build_preprocessing_pipeline(X_train)
    X_train_prep, X_test_prep = fit_transform_data(preprocessing, X_train, X_test)

    # 4. Random Forest
    _section("Random Forest")
    best_rf = train_random_forest(X_train_prep, y_train)
    preds_rf = best_rf.predict(X_test_prep)
    evaluate_predictions(y_test, preds_rf, y_train, label="Random Forest")

    # 5. MLP (PyTorch)
    _section("MLP (PyTorch)")
    mlp_model, tensors = train_mlp(X_train_prep, X_test_prep, y_train, y_test)
    preds_mlp = predict_mlp(mlp_model, tensors)
    preds_mlp_np = preds_mlp.numpy().ravel()
    evaluate_predictions(y_test, preds_mlp_np, y_train, label="MLP")

    # 6. Comparison
    _section("Comparison")
    plot_model_comparison(y_test.values, preds_rf, preds_mlp_np)

    print("\nTop errors - Random Forest:")
    print(top_errors(y_test.values, preds_rf))
    print("\nTop errors - MLP:")
    print(top_errors(y_test.values, preds_mlp_np))


if __name__ == "__main__":
    main()
