# main_vae.py

import pandas as pd
from data_utils_vae import load_and_preprocess_data, get_training_testing_data
from training_utils_vae import run_iteration

def main():
    """
    Main driver script to run the hyperparameter tuning experiment.
    """
    print("Loading and preprocessing the full dataset once...")
    full_df = load_and_preprocess_data()
    
    # --- Define the Hyperparameter Tuning Experiments ---
    experiments = [
        {'window': 60, 'quantile': 0.995}, # The conservative baseline
        {'window': 12, 'quantile': 0.99},  # The sensitive baseline from last attempt
        {'window': 12, 'quantile': 0.98},  # Even more sensitive threshold
        {'window': 6,  'quantile': 0.99},  # Even smaller window
        {'window': 6,  'quantile': 0.98}   # Most sensitive combination
    ]
    
    best_f1 = -1
    best_params = None
    results = []

    # Just using the first iteration for a focused tuning experiment
    iteration_name = "Train until April"
    train_end_date = "2020-04-01"
    
    X_train, X_test, y_test, test_timestamps = get_training_testing_data(full_df, train_end_date)
    
    for params in experiments:
        window = params['window']
        quantile = params['quantile']
        
        # Run the cycle and get the F1 score
        f1 = run_iteration(X_train, X_test, y_test, test_timestamps, iteration_name, window, quantile)
        results.append({'params': params, 'f1_score': f1})
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            
    # --- Print the Final Summary of the Experiment ---
    print("\n\n----- Hyperparameter Tuning Experiment Summary -----")
    print("F1 Scores for each parameter combination:")
    for res in results:
        print(f"- Params: {res['params']} | F1-Score: {res['f1_score']:.4f}")
        
    print("\n--- Best Performing Parameters ---")
    print(f"Best Params: {best_params}")
    print(f"Best Anomaly F1-Score: {best_f1:.4f}")
        
    print("\nExperiment finished.")

if __name__ == "__main__":
    main()