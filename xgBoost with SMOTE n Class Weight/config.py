# config.py

DATA_PATH = "Industrial_fault_detection.csv"

TEST_SIZE = 0.3
RANDOM_STATE = 42

XGBOOST_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 4,  # Classes: 0, 1, 2, 3
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss'
}
