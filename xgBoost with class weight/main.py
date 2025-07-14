# main.py 

from preprocessing import load_and_preprocess_data
from evaluate_model import evaluate_model
from tune_model import tune_xgboost

def main():
    print("📥 Loading and preprocessing data (with class weights)...")
    X_train, X_test, y_train, y_test, class_weight_dict = load_and_preprocess_data()

    print("🧪 Hyperparameter tuning for XGBoost...")
    model = tune_xgboost(X_train, y_train, class_weight_dict)

    print("📈 Evaluating model performance...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()