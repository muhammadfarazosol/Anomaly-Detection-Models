from data_utils import load_and_preprocess_data, get_training_testing_data
from training_utils import run_iteration

def main():
    print("Loading and preprocessing the full dataset once...")
    full_df = load_and_preprocess_data()

    iterations = {
        "Train until April": "2020-04-01",
        "Train until May": "2020-05-01",
        "Train until June": "2020-06-01",
        "Train until July": "2020-07-01",
    }

    results = {}
    for name, cutoff in iterations.items():
        X_train, X_test, y_test, ts = get_training_testing_data(full_df, cutoff, seq_len=10)
        f1 = run_iteration(X_train, X_test, y_test, ts, name)
        results[name] = f1

    print("\n\n----- Overall Project Summary -----")
    for name, score in results.items():
        print(f"- {name}: {score:.4f}")
    print("\nAll iterations complete. Project finished.")

if __name__ == "__main__":
    main()
