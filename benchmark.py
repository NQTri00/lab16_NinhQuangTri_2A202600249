import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import time
import json
import os

def run_benchmark():
    # 1. Load Data
    print("--- Step 1: Loading Data ---")
    data_path = 'creditcard.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Ensure you have downloaded it from Kaggle.")
        return

    start_load = time.time()
    df = pd.read_csv(data_path)
    load_time = time.time() - start_load
    print(f"Data loaded: {len(df)} rows in {load_time:.2f}s")

    # 2. Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training
    print("\n--- Step 2: Training LightGBM ---")
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    start_train = time.time()
    # Using lgb.train with early stopping or fixed rounds
    # We'll use a fixed number of rounds for benchmark consistency
    bst = lgb.train(params, train_data, num_boost_round=100)
    train_time = time.time() - start_train
    print(f"Training completed in {train_time:.2f}s")

    # 4. Evaluation
    print("\n--- Step 3: Evaluation ---")
    y_pred_prob = bst.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    best_iteration = bst.best_iteration

    # 5. Inference Latency (1 row)
    print("\n--- Step 4: Inference Benchmarking ---")
    single_row = X_test.iloc[[0]]
    # Warmup
    for _ in range(10): bst.predict(single_row)
    
    start_inf_1 = time.time()
    iters = 100
    for _ in range(iters):
        bst.predict(single_row)
    inf_latency_1 = (time.time() - start_inf_1) / iters

    # 6. Inference Throughput (1000 rows)
    batch_rows = X_test.iloc[:1000]
    start_inf_1000 = time.time()
    bst.predict(batch_rows)
    inf_latency_1000 = time.time() - start_inf_1000

    results = {
        "load_time": f"{load_time:.4f}s",
        "train_time": f"{train_time:.4f}s",
        "best_iteration": best_iteration,
        "auc_roc": f"{auc_roc:.4f}",
        "accuracy": f"{accuracy:.4f}",
        "f1_score": f"{f1:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "inf_latency_1_row": f"{inf_latency_1*1000:.4f}ms",
        "inf_throughput_1000_rows": f"{inf_latency_1000*1000:.4f}ms"
    }

    print("\nBenchmark Results Summary:")
    print(json.dumps(results, indent=4))

    # Save to JSON
    with open('benchmark_result.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nMetrics saved to benchmark_result.json")

if __name__ == "__main__":
    run_benchmark()
