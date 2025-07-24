import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('data/Industrial_fault_detection.csv')
print("Dataset shape:", df.shape)

# ➤ Convert multi-class to binary (0 = normal, 1 = any fault)
df['Fault_Label'] = df['Fault_Type'].apply(lambda x: 0 if x == 0 else 1)
print("Binary class distribution:\n", df['Fault_Label'].value_counts())

# ➤ Features and binary label
X = df.drop(['Fault_Type', 'Fault_Label'], axis=1)
y = df['Fault_Label']

# ➤ Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ➤ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# ➤ Apply SMOTE on training set only
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ➤ Split training set for early stopping
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_sm, y_train_sm, test_size=0.1, stratify=y_train_sm, random_state=42)

# ➤ Focal Loss
def focal_binary(alpha=0.75, gamma=2.0):
    def fl_obj(preds, dtrain):
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # sigmoid
        preds = np.clip(preds, 1e-9, 1 - 1e-9)

        # Vectorized focal loss gradient & hessian
        p_t = preds * labels + (1 - preds) * (1 - labels)
        grad = alpha * (preds - labels) * ((1 - p_t) ** gamma)
        hess = alpha * ((1 - p_t) ** gamma) * (
            preds * (1 - preds) + gamma * (preds - labels) ** 2
        )

        return grad, hess
    return fl_obj

# ➤ DMatrix
dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# ➤ Parameters
params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.5,
    'eval_metric': 'logloss'
}

# ➤ Train model
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    obj=focal_binary(alpha=0.75, gamma=2.0),
    evals=[(dval, 'eval')],
    early_stopping_rounds=15,
    verbose_eval=True
)

# ➤ Evaluate
y_train_pred = (model.predict(dtrain) > 0.5).astype(int)
y_test_pred = (model.predict(dtest) > 0.5).astype(int)

train_acc = accuracy_score(y_train_final, y_train_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100
auc = roc_auc_score(y_test, model.predict(dtest))

print(f"\n✅ Train Accuracy: {train_acc:.2f}%")
print(f"✅ Test Accuracy: {test_acc:.2f}%")
print(f"✅ ROC AUC Score: {auc:.4f}\n")

print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred))

# ➤ Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Binary Fault Detection)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
# plt.show()
