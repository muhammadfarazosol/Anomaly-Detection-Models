from sklearn.metrics import accuracy_score, classification_report, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"🎯 Macro F1-score: {macro_f1 * 100:.2f}%\n")

    print("📋 Classification Report (per class):")
    print(classification_report(y_test, y_pred, zero_division=0))
