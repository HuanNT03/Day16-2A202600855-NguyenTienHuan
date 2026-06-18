import pandas as pd
import numpy as np
import time
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_benchmark():
    print("=== STARTING LIGHTGBM BENCHMARK ON CPU ===")
    
    # 1. Measure Data Loading Time
    start_time = time.time()
    df = pd.read_csv("creditcard.csv")
    load_time = time.time() - start_time
    print(f"Data load time: {load_time:.4f} seconds")
    print(f"Dataset shape: {df.shape}")
    
    # Preprocessing
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Define parameters (optimized for CPU running on 8 vCPUs of r5.2xlarge)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1  # Use all 8 cores
    }
    
    # 2. Measure Training Time
    print("Training LightGBM model...")
    start_time = time.time()
    
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    train_time = time.time() - start_time
    best_iteration = model.best_iteration
    print(f"Training completed in: {train_time:.4f} seconds")
    print(f"Best iteration: {best_iteration}")
    
    # 3. Model Predictions and Metrics Evaluation
    y_pred_prob = model.predict(X_test, num_iteration=best_iteration)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    
    print("\n=== CLASSIFICATION METRICS ===")
    print(f"AUC-ROC: {auc_roc:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"F1-Score: {f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    
    # 4. Measure Inference Latency (Single Row)
    print("\nMeasuring Inference Latency (1 row)...")
    single_row = X_test.iloc[[0]]
    
    latencies = []
    # Warmup
    for _ in range(50):
        _ = model.predict(single_row, num_iteration=best_iteration)
        
    for _ in range(1000):
        t0 = time.time()
        _ = model.predict(single_row, num_iteration=best_iteration)
        latencies.append(time.time() - t0)
        
    avg_latency_ms = np.mean(latencies) * 1000
    print(f"Average latency for 1 row: {avg_latency_ms:.4f} ms")
    
    # 5. Measure Inference Throughput (1000 Rows)
    print("Measuring Inference Throughput (1000 rows)...")
    batch_rows = X_test.iloc[:1000]
    
    batch_times = []
    # Warmup
    for _ in range(10):
        _ = model.predict(batch_rows, num_iteration=best_iteration)
        
    for _ in range(100):
        t0 = time.time()
        _ = model.predict(batch_rows, num_iteration=best_iteration)
        batch_times.append(time.time() - t0)
        
    avg_batch_time = np.mean(batch_times)
    throughput = 1000 / avg_batch_time
    print(f"Average time for 1000 rows batch: {avg_batch_time * 1000:.4f} ms")
    print(f"Inference throughput: {throughput:.2f} rows/second")
    
    # Save results to benchmark_result.json
    results = {
        "data_load_time_seconds": round(load_time, 4),
        "training_time_seconds": round(train_time, 4),
        "best_iteration": best_iteration,
        "auc_roc": round(auc_roc, 6),
        "accuracy": round(accuracy, 6),
        "f1_score": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "inference_latency_1_row_ms": round(avg_latency_ms, 4),
        "inference_throughput_1000_rows_per_second": round(throughput, 2)
    }
    
    with open("benchmark_result.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved benchmark results to 'benchmark_result.json'")

if __name__ == "__main__":
    run_benchmark()
