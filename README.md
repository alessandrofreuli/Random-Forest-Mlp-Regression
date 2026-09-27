# Random Forest & MLP Regression

A project to predict the **Skill Retention Score** from the `ai_student_impact_dataset.csv` dataset, comparing two models:

- **Random Forest** (scikit-learn, with `GridSearchCV`)
- **MLP** (feed-forward neural network in PyTorch)

## Project structure

```
random-forest-mlp-regression/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   └── ai_student_impact_dataset.csv   # add manually
├── notebooks/
│   └── skill_retention.ipynb           # original exploration notebook
├── src/
│   └── rf_mlp_regression/
│       ├── config.py           # paths and hyperparameters
│       ├── data.py             # dataset loading and splitting
│       ├── preprocessing.py    # numeric/categorical pipelines
│       ├── models/
│       │   ├── random_forest.py
│       │   └── mlp.py
│       ├── evaluate.py         # RMSE, relative error, top errors
│       └── visualize.py        # plots
└── main.py                     # runs the full pipeline
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # installs the rf_mlp_regression package in editable mode
```

Place the dataset at `data/ai_student_impact_dataset.csv`.

## Run

Full pipeline (preprocessing → RF and MLP training → evaluation → comparison):

```bash
python main.py
```

## Original notebook

The notebook `notebooks/skill_retention.ipynb` is kept for interactive data exploration and plotting (correlations, histograms, scatter plots), while training/evaluation logic now lives in the `src/rf_mlp_regression` package for reuse and testability.
