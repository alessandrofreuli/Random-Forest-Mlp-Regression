# Random Forest & MLP Regression

Predicts students' **Skill Retention Score** after a semester of generative-AI use, comparing a tuned **Random Forest** (scikit-learn) with a feed-forward **MLP** (PyTorch) on the same data and preprocessing.

The dataset is synthetic: 50,000 students described by major, year, GPA, weekly GenAI hours, prompt-engineering skill, institutional AI policy, exam anxiety and more: https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students

## Features

- **Fair comparison**: same train/test split and preprocessing for both models.
- **Random Forest**: tuned with `GridSearchCV` (3-fold CV, negative RMSE).
- **PyTorch MLP**: trained on a standardised target, rescaled back at prediction time.
- **Leak-aware**: `Post_Semester_GPA` and `Student_ID` are dropped.
- **Single config file**: paths, split, grid and MLP hyperparameters in `config.py`.

## How it works

1. **Load** the CSV, drop `Student_ID` and `Post_Semester_GPA`.
2. **Split** train/test (`test_size=0.2`, `random_state=42`).
3. **Preprocess** (fit on train only):
   - numeric/boolean: median imputation → `StandardScaler`
   - categorical: most-frequent imputation → `OneHotEncoder`
4. **Random Forest**: grid over `n_estimators ∈ {100, 200}` and `max_features ∈ {4, 6, 8, 10}`.
5. **MLP**: `Linear(in, 50) → ReLU → Linear(50, 40) → ReLU → Linear(40, 1)`, MSE loss, SGD (lr 0.01, batch 128, 25 epochs).
6. **Evaluate**: RMSE, relative error, scatter plots and the top-5 worst predictions per model.

## Results

| Model | RMSE | Relative error |
| --- | ---: | ---: |
| Random Forest (`n_estimators=200`, `max_features=8`) | 11.87 | 13.30 % |
| MLP (50-40, SGD) | 11.81 | 13.24 % |

The two models are practically tied. With an error of about 12 points on a ~89-point target range, most of the remaining error is likely noise in the synthetic data rather than a limit of model capacity. Exact numbers may vary slightly with library versions and hardware.

## Project structure

```
random-forest-mlp-regression/
├── main.py                        # runs the full pipeline
├── pyproject.toml
├── requirements.txt
├── data/
│   └── ai_student_impact_dataset.csv
├── notebooks/
│   └── skill_retention.ipynb      # exploration (comments in Italian)
└── src/rf_mlp_regression/
    ├── config.py                  # paths, split, grid, MLP hyperparameters
    ├── data.py                    # loading and split
    ├── preprocessing.py           # ColumnTransformer
    ├── evaluate.py                # RMSE, relative error, top errors
    ├── visualize.py               # plots and model comparison
    └── models/
        ├── random_forest.py
        └── mlp.py
```

## Installation

Requires **Python ≥ 3.9**.

```bash
git clone https://github.com/alessandrofreuli/random-forest-mlp-regression.git
cd random-forest-mlp-regression

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
python main.py
```

For interactive exploration (correlations, histograms, scatter plots), open `notebooks/skill_retention.ipynb`.
