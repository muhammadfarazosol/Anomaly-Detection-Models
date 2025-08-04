# # engine_fault_detector.py

# import pandas as pd
# import numpy as np
# from collections import Counter
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, accuracy_score, f1_score
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.combine import SMOTEENN

# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# class EngineFaultDetector:
#     def __init__(self, random_state=42):
#         self.random_state = random_state
#         self.models = {}

#     def load_data(self, path):
#         print("🔍 Loading Dataset...")
#         self.df = pd.read_csv(path)
#         print(f"✅ Data loaded from {path}")
#         print(f"📊 Dataset shape: {self.df.shape}")

#     def engineer_features(self):
#         print("\n🔧 Engineering Additional Features...")
#         df = self.df.copy()

#         # Example feature engineering
#         df['Sensor_Mean'] = df.iloc[:, :-1].mean(axis=1)
#         df['Sensor_Std'] = df.iloc[:, :-1].std(axis=1)
#         df['Sensor_Max'] = df.iloc[:, :-1].max(axis=1)
#         df['Sensor_Min'] = df.iloc[:, :-1].min(axis=1)
#         df['Sensor_Range'] = df['Sensor_Max'] - df['Sensor_Min']
#         df['Sensor_Zero_Count'] = (df.iloc[:, :-1] == 0).sum(axis=1)
#         df['Sensor_NonZero_Ratio'] = (df.iloc[:, :-1] != 0).sum(axis=1) / df.iloc[:, :-1].shape[1]
#         df['Sensor_Skew'] = df.iloc[:, :-1].skew(axis=1)

#         self.df = df
#         print(f"✅ Added 8 engineered features. New shape: {df.shape}")

#     def prepare_data(self):
#         print("\n🔧 Advanced Data Preparation...")
#         df = self.df.copy()
#         X = df.drop('Engine_Condition', axis=1)
#         y = df['Engine_Condition']

#         X_temp, self.X_test, y_temp, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.random_state)
#         self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=self.random_state)

#         print("Original class distribution:")
#         print("Train:", Counter(self.y_train))
#         print("Val:", Counter(self.y_val))
#         print("Test:", Counter(self.y_test))

#         self.scaler = StandardScaler()
#         self.X_train_scaled = self.scaler.fit_transform(self.X_train)
#         self.X_val_scaled = self.scaler.transform(self.X_val)
#         self.X_test_scaled = self.scaler.transform(self.X_test)

#         # SMOTEENN instead of SMOTE
#         print("\n⚖️ Applying SMOTEENN sampling strategy...")
#         smoteenn = SMOTEENN(random_state=self.random_state)
#         self.X_train_scaled, self.y_train_resampled = smoteenn.fit_resample(self.X_train_scaled, self.y_train)
#         print("After SMOTEENN:", Counter(self.y_train_resampled))
#         print("✅ Data preparation completed!")

#     def compute_sample_weights(self):
#         weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train_resampled), y=self.y_train_resampled)
#         self.class_weights_dict = dict(zip(np.unique(self.y_train_resampled), weights))
#         self.sample_weights = np.array([self.class_weights_dict[y] for y in self.y_train_resampled])

#     def tune_xgboost_balanced(self):
#         print("\n⚙️ Tuning XGBoost_Balanced Hyperparameters...")
#         param_grid = {
#             'n_estimators': [150, 200],
#             'max_depth': [4, 5],
#             'learning_rate': [0.05, 0.1],
#             'subsample': [0.9],
#             'colsample_bytree': [0.9],
#             'gamma': [0, 0.1],
#             'reg_alpha': [0.5],
#             'reg_lambda': [1]
#         }

#         xgb_model = XGBClassifier(
#             objective='multi:softprob',
#             eval_metric='mlogloss',
#             random_state=self.random_state,
#             n_jobs=-1,
#             # use_label_encoder=False
#         )

#         grid = GridSearchCV(
#             estimator=xgb_model,
#             param_grid=param_grid,
#             scoring='f1_macro',
#             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
#             verbose=1,
#             n_jobs=-1
#         )

#         grid.fit(self.X_train_scaled, self.y_train_resampled, sample_weight=self.sample_weights)

#         print(f"\n✅ Best parameters: {grid.best_params_}")
#         print(f"✅ Best cross-validation score (Macro F1): {grid.best_score_:.4f}")

#         self.models['XGBoost_Balanced'] = grid.best_estimator_

#     def train_other_models(self):
#         print("\n🚀 Training Multiple Models...")

#         self.models['XGBoost_Conservative'] = XGBClassifier(
#             max_depth=3, n_estimators=100, reg_alpha=1.0, reg_lambda=1.0,
#             learning_rate=0.05, objective='multi:softprob',
#             random_state=self.random_state, n_jobs=-1)
#         self.models['XGBoost_Conservative'].fit(self.X_train_scaled, self.y_train_resampled, sample_weight=self.sample_weights)

#         self.models['LightGBM'] = LGBMClassifier(
#             n_estimators=200, max_depth=6, class_weight='balanced', random_state=self.random_state)
#         self.models['LightGBM'].fit(self.X_train_scaled, self.y_train_resampled)

#         self.models['RandomForest'] = RandomForestClassifier(
#             n_estimators=200, class_weight='balanced', random_state=self.random_state)
#         self.models['RandomForest'].fit(self.X_train_scaled, self.y_train_resampled)

#         print("✅ All models trained successfully!")

#     def evaluate_all_models(self):
#         print("\n📊 Evaluating All Models...")
#         for name, model in self.models.items():
#             y_pred = model.predict(self.X_test_scaled)
#             acc = accuracy_score(self.y_test, y_pred)
#             macro_f1 = f1_score(self.y_test, y_pred, average='macro')
#             val_pred = model.predict(self.X_val_scaled)
#             val_acc = accuracy_score(self.y_val, val_pred)
#             overfit_gap = abs(val_acc - acc)

#             print(f"\n🔍 Evaluating {name}...")
#             print(f"   Test Accuracy: {acc:.4f} ({acc * 100:.1f}%)")
#             print(f"   Macro F1: {macro_f1:.4f}")
#             print(f"   Overfitting Gap: {overfit_gap:.4f}")
#             print(classification_report(self.y_test, y_pred, digits=3))

#     def run_complete_analysis(self, file_path):
#         self.load_data(file_path)
#         self.engineer_features()
#         self.prepare_data()
#         self.compute_sample_weights()
#         self.tune_xgboost_balanced()
#         self.train_other_models()
#         self.evaluate_all_models()

# if __name__ == "__main__":
#     detector = EngineFaultDetector()
#     detector.run_complete_analysis('data/engine_fault_detection_dataset.csv')

# engine_fault_detector.py
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek 

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class EngineFaultDetector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}

    def load_data(self, path):
        print("🔍 Loading Dataset...")
        self.df = pd.read_csv(path)
        print(f"✅ Data loaded from {path}")
        print(f"📊 Dataset shape: {self.df.shape}")

    def engineer_features(self):
        print("\n🔧 Engineering Additional Features...")
        df = self.df.copy()

        # Feature engineering
        df['Sensor_Mean'] = df.iloc[:, :-1].mean(axis=1)
        df['Sensor_Std'] = df.iloc[:, :-1].std(axis=1)
        df['Sensor_Max'] = df.iloc[:, :-1].max(axis=1)
        df['Sensor_Min'] = df.iloc[:, :-1].min(axis=1)
        df['Sensor_Range'] = df['Sensor_Max'] - df['Sensor_Min']
        df['Sensor_Zero_Count'] = (df.iloc[:, :-1] == 0).sum(axis=1)
        df['Sensor_NonZero_Ratio'] = (df.iloc[:, :-1] != 0).sum(axis=1) / df.iloc[:, :-1].shape[1]
        df['Sensor_Skew'] = df.iloc[:, :-1].skew(axis=1)

        self.df = df
        print(f"✅ Added 8 engineered features. New shape: {df.shape}")

    def prepare_data(self):
        print("\n🔧 Advanced Data Preparation...")
        df = self.df.copy()
        X = df.drop('Engine_Condition', axis=1)
        y = df['Engine_Condition']

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.random_state)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=self.random_state)

        print("Original class distribution:")
        print("Train:", Counter(self.y_train))
        print("Val:", Counter(self.y_val))
        print("Test:", Counter(self.y_test))

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # SMOTETomek
        print("\n⚖️ Applying SMOTETomek sampling strategy...")
        smotetomek = SMOTETomek(random_state=self.random_state)
        self.X_train_scaled, self.y_train_resampled = smotetomek.fit_resample(self.X_train_scaled, self.y_train)
        print("After SMOTETomek:", Counter(self.y_train_resampled))
        print("✅ Data preparation completed!")

    def compute_sample_weights(self):
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train_resampled), y=self.y_train_resampled)
        self.class_weights_dict = dict(zip(np.unique(self.y_train_resampled), weights))
        self.sample_weights = np.array([self.class_weights_dict[y] for y in self.y_train_resampled])

    def tune_xgboost_balanced(self):
        print("\n⚙️ Tuning XGBoost_Balanced Hyperparameters...")
        param_grid = {
            'n_estimators': [150, 200],
            'max_depth': [4, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.9],
            'colsample_bytree': [0.9],
            'gamma': [0, 0.1],
            'reg_alpha': [0.5],
            'reg_lambda': [1]
        }

        xgb_model = XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1,
            use_label_encoder=False # Set to False to suppress the warning
        )

        grid = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            verbose=1,
            n_jobs=-1
        )

        grid.fit(self.X_train_scaled, self.y_train_resampled, sample_weight=self.sample_weights)

        print(f"\n✅ Best parameters: {grid.best_params_}")
        print(f"✅ Best cross-validation score (Macro F1): {grid.best_score_:.4f}")

        self.models['XGBoost_Balanced'] = grid.best_estimator_

    def train_other_models(self):
        print("\n🚀 Training Multiple Models...")

        self.models['XGBoost_Conservative'] = XGBClassifier(
            max_depth=3, n_estimators=100, reg_alpha=1.0, reg_lambda=1.0,
            learning_rate=0.05, objective='multi:softprob',
            random_state=self.random_state, n_jobs=-1,
            use_label_encoder=False # Set to False to suppress the warning
        )
        self.models['XGBoost_Conservative'].fit(self.X_train_scaled, self.y_train_resampled, sample_weight=self.sample_weights)

        self.models['LightGBM'] = LGBMClassifier(
            n_estimators=200, max_depth=6, class_weight='balanced', random_state=self.random_state)
        self.models['LightGBM'].fit(self.X_train_scaled, self.y_train_resampled)

        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=self.random_state)
        self.models['RandomForest'].fit(self.X_train_scaled, self.y_train_resampled)

        print("✅ All models trained successfully!")

    def evaluate_all_models(self):
        print("\n📊 Evaluating All Models...")
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test_scaled)
            acc = accuracy_score(self.y_test, y_pred)
            macro_f1 = f1_score(self.y_test, y_pred, average='macro')
            val_pred = model.predict(self.X_val_scaled)
            val_acc = accuracy_score(self.y_val, val_pred)
            overfit_gap = abs(val_acc - acc)

            print(f"\n🔍 Evaluating {name}...")
            print(f"   Test Accuracy: {acc:.4f} ({acc * 100:.1f}%)")
            print(f"   Macro F1: {macro_f1:.4f}")
            print(f"   Overfitting Gap: {overfit_gap:.4f}")
            print(classification_report(self.y_test, y_pred, digits=3))

    def run_complete_analysis(self, file_path):
        self.load_data(file_path)
        self.engineer_features()
        self.prepare_data()
        self.compute_sample_weights()
        self.tune_xgboost_balanced()
        self.train_other_models()
        self.evaluate_all_models()

detector = EngineFaultDetector()
detector.run_complete_analysis('data/engine_fault_detection_dataset.csv')