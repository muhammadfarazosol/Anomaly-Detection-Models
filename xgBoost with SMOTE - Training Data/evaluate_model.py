# evaluate_model.py

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy * 100:.2f}%\n")

    print("📋 Classification Report (per class):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # print("🔷 Confusion Matrix:")
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(4), yticklabels=range(4))
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")
    # plt.show()
