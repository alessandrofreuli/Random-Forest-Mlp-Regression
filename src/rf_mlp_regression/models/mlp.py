"""PyTorch MLP model for regression."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .. import config


def prepare_tensors(X_train_prep, X_test_prep, y_train, y_test):
    """Convert data to PyTorch tensors and normalize the target."""
    X_train_t = torch.FloatTensor(X_train_prep.copy())
    X_test_t = torch.FloatTensor(X_test_prep.copy())
    y_train_t = torch.FloatTensor(y_train.values.copy()).view(-1, 1)
    y_test_t = torch.FloatTensor(y_test.values.copy()).view(-1, 1)

    y_mean = y_train_t.mean()
    y_std = y_train_t.std()
    y_train_s = (y_train_t - y_mean) / y_std
    y_test_s = (y_test_t - y_mean) / y_std

    train_dataset = TensorDataset(X_train_t, y_train_s)
    train_loader = DataLoader(
        train_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True
    )

    return {
        "X_train_t": X_train_t,
        "X_test_t": X_test_t,
        "y_train_s": y_train_s,
        "y_test_s": y_test_s,
        "y_mean": y_mean,
        "y_std": y_std,
        "train_loader": train_loader,
    }


def build_model(input_dim: int) -> nn.Sequential:
    """Build the MLP architecture."""
    h1, h2 = config.MLP_HIDDEN_SIZES
    return nn.Sequential(
        nn.Linear(input_dim, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
    )


def train_minibatch(model, train_loader, opt, loss_fn, n_epochs=None):
    """Mini-batch training loop."""
    n_epochs = n_epochs or config.MLP_EPOCHS
    for epoch in range(n_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, loss: {avg_loss:.4f}")
    return model


def train_mlp(X_train_prep, X_test_prep, y_train, y_test):
    """Prepare data, build and train the MLP end-to-end."""
    torch.manual_seed(config.RANDOM_STATE)

    tensors = prepare_tensors(X_train_prep, X_test_prep, y_train, y_test)
    model = build_model(tensors["X_train_t"].shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=config.MLP_LR)
    loss_fn = nn.MSELoss()

    train_minibatch(model, tensors["train_loader"], optimizer, loss_fn)

    return model, tensors


def predict(model, tensors):
    """Predict on the test set and rescale back to the original range."""
    with torch.no_grad():
        preds_s = model(tensors["X_test_t"])
    preds = preds_s * tensors["y_std"] + tensors["y_mean"]
    return preds
