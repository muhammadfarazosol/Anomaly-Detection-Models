# tune_model.py 

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from config import RANDOM_STATE
import warnings
warnings.filterwarnings("ignore")

def tune_xgboost(X_train, y_train, class_weight_dict):
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 150, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        scale_pos_weight=None 
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    print("🔍 Running grid search (this may take a while)...")
    # Pass class weights via sample_weight
    class_weighted = [class_weight_dict[y] for y in y_train]
    grid_search.fit(X_train, y_train, sample_weight=class_weighted)

    print(f"\n🏁 Best Parameters: {grid_search.best_params_}")
    print(f"✅ Best CV Accuracy: {grid_search.best_score_ * 100:.2f}%")

    return grid_search.best_estimator_

