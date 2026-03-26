from src.rf_model import train_rf
from src.gnn_model import train_gnn
from src.utils import load_data
from src.hybrid_model import adaptive_hybrid

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import time
import numpy as np


def evaluate_early_detection(X, y, config):
    from sklearn.model_selection import train_test_split
    from src.rf_model import train_rf

    print("\n🚀 Early Detection Analysis")

    sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    for size in sizes:
        print(f"\n--- Using {int(size*100)}% of data ---")

        subset_size = int(len(X) * size)
        X_sub = X[:subset_size]
        y_sub = y[:subset_size]

        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )

        acc, _, _ = train_rf(X_train_sub, X_test_sub, y_train_sub, y_test_sub)

        print(f"Accuracy at {int(size*100)}% data:", acc)


def run_training(config):
    print("🔥 Running Training Pipeline...\n")

    # Load data
    X_train, X_test, y_train, y_test = load_data(config)

    # 🔥 NOVELTY: Early Detection
    evaluate_early_detection(X_train, y_train, config)

    # -------------------
    # Random Forest (with time)
    # -------------------
    start = time.time()
    rf_acc, rf_preds, rf_probs = train_rf(X_train, X_test, y_train, y_test)
    rf_time = time.time() - start

    # -------------------
    # GNN (with time)
    # -------------------
    start = time.time()
    gnn_acc, gnn_preds, gnn_probs = train_gnn(
        X_train, X_test, y_train, y_test, config
    )
    gnn_time = time.time() - start

    # -------------------
    # 🔥 Adaptive Hybrid
    # -------------------
    print("\n⚡ Running Adaptive Hybrid Model...")

    hybrid_preds, hybrid_conf = adaptive_hybrid(
        rf_preds, gnn_probs, threshold=0.9
    )

    hybrid_acc = accuracy_score(y_test, hybrid_preds)

    print("⚡ Hybrid Accuracy:", hybrid_acc)

    # -------------------
    # 🔥 Confidence Analysis
    # -------------------
    print("\n📊 Confidence Analysis:")
    print("Average Confidence:", hybrid_conf.mean())
    print("High Confidence Predictions:", (hybrid_conf > 0.9).sum())

    # -------------------
    # 🚨 False Positive Rate
    # -------------------
    cm = confusion_matrix(y_test, hybrid_preds)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)

    print("\n🚨 False Positive Rate:", fpr)

    # -------------------
    # 📈 ROC-AUC
    # -------------------
    try:
        auc = roc_auc_score(y_test, rf_probs[:, 1])
        print("\n📈 ROC-AUC Score (RF):", auc)
    except:
        print("\n⚠ ROC-AUC could not be calculated")

    # -------------------
    # 📊 Stability
    # -------------------
    stability = np.std([rf_acc, gnn_acc, hybrid_acc])
    print("\n📊 Model Stability (Std Dev):", stability)

    # -------------------
    # ⏱ Detection Time
    # -------------------
    print("\n⏱ Detection Time:")
    print("RF Time:", rf_time)
    print("GNN Time:", gnn_time)

    # -------------------
    # Metrics
    # -------------------
    print("\n📈 RF Metrics:")
    print(classification_report(y_test, rf_preds))

    print("\n📈 GNN Metrics:")
    print(classification_report(y_test, gnn_preds))

    print("\n📈 Hybrid Metrics:")
    print(classification_report(y_test, hybrid_preds))

    # -------------------
    # Final comparison
    # -------------------
    print("\n📊 FINAL COMPARISON")
    print("-------------------")
    print(f"RF Accuracy     : {rf_acc}")
    print(f"GNN Accuracy    : {gnn_acc}")
    print(f"Hybrid Accuracy : {hybrid_acc}")
    