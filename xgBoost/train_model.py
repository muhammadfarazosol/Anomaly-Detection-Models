# train_model.py

from xgboost import XGBClassifier
from config import XGBOOST_PARAMS

def train_xgboost(X_train, y_train):
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    return model
