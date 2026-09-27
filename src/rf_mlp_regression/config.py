"""Global project configuration."""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "ai_student_impact_dataset.csv"

# Columns to drop from the raw dataset
DROP_COLUMNS = ["Student_ID", "Post_Semester_GPA"]

# Regression target
TARGET = "Skill_Retention_Score"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Random Forest / GridSearch
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_features": [4, 6, 8, 10],
}
RF_CV_FOLDS = 3

# MLP
MLP_HIDDEN_SIZES = (50, 40)
MLP_BATCH_SIZE = 128
MLP_LR = 0.01
MLP_EPOCHS = 25
