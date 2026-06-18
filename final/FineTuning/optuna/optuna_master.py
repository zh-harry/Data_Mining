import optuna
import subprocess
import re
import os

# 設定絕對路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

KNN_DIR = os.path.join(ROOT_DIR, "classification", "KNN")
SVM_DIR = os.path.join(ROOT_DIR, "classification", "SVM")
DBSCAN_DIR = os.path.join(ROOT_DIR, "clustering", "DBSCAN")
KMEANS_DIR = os.path.join(ROOT_DIR, "clustering", "kmeans")

# 資料隔離設定
TRAIN_CSV = os.path.join(ROOT_DIR, "data", "train_mock_std_pca.csv")
VAL_CSV = os.path.join(ROOT_DIR, "data", "val_mock_std_pca.csv")     

# 測試集僅用於最終效能評估
TEST_CSV = os.path.join(ROOT_DIR, "data", "dry_bean_test_std_pca.csv")

KNN_PRED_CSV = os.path.join(KNN_DIR, "knn_prediction.csv")
SVM_PRED_CSV = os.path.join(SVM_DIR, "svm_prediction.csv")

def execute_pipeline(class_cmd, cluster_cmd, cwd_class, cwd_cluster):
    """共用的執行管線函式 (靜默模式)"""
    try:
        subprocess.run(class_cmd, cwd=cwd_class, capture_output=True, text=True, check=True)

        exe_absolute_path = os.path.join(cwd_cluster, "main.exe")
        cluster_cmd[0] = exe_absolute_path
        
        cluster_result = subprocess.run(cluster_cmd, cwd=cwd_cluster, capture_output=True, text=True, check=True)
        
        match = re.search(r"Overall Accuracy:\s*([0-9.]+)", cluster_result.stdout)
        if match:
            return float(match.group(1))
        return 0.0

    except subprocess.CalledProcessError:
        return 0.0

# 1. KNN + DBSCAN 組合
def objective_knn_dbscan(trial):
    k_knn = trial.suggest_int("k_knn", 5, 50)
    threshold = trial.suggest_float("threshold", 0.6, 0.99)
    eps = trial.suggest_float("eps", 0.1, 5.0)
    min_samples = trial.suggest_int("min_samples", 5, 50)

    # 確保模型評估使用 VAL_CSV
    class_cmd = ["python", "knn.py", TRAIN_CSV, VAL_CSV, str(k_knn), str(threshold)]
    cluster_cmd = ["main.exe", KNN_PRED_CSV, VAL_CSV, str(eps), str(min_samples)]
    
    return execute_pipeline(class_cmd, cluster_cmd, KNN_DIR, DBSCAN_DIR)

# 2. SVM + DBSCAN 組合
def objective_svm_dbscan(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
    threshold = trial.suggest_float("threshold", 0.6, 0.99)
    eps = trial.suggest_float("eps", 0.1, 5.0)
    min_samples = trial.suggest_int("min_samples", 5, 50)

    # 確保模型評估使用 VAL_CSV
    class_cmd = ["python", "svm.py", TRAIN_CSV, VAL_CSV, kernel, str(threshold)]
    cluster_cmd = ["main.exe", SVM_PRED_CSV, VAL_CSV, str(eps), str(min_samples)]
    
    return execute_pipeline(class_cmd, cluster_cmd, SVM_DIR, DBSCAN_DIR)

# 3. KNN + KMEANS 組合
def objective_knn_kmeans(trial):
    k_knn = trial.suggest_int("k_knn", 5, 50)
    threshold = trial.suggest_float("threshold", 0.6, 0.99)
    k_clusters = trial.suggest_int("k_clusters", 2, 20)

    # 確保模型評估使用 VAL_CSV
    class_cmd = ["python", "knn.py", TRAIN_CSV, VAL_CSV, str(k_knn), str(threshold)]
    cluster_cmd = ["main.exe", KNN_PRED_CSV, VAL_CSV, str(k_clusters)]
    
    return execute_pipeline(class_cmd, cluster_cmd, KNN_DIR, KMEANS_DIR)

# 4. SVM + KMEANS 組合
def objective_svm_kmeans(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
    threshold = trial.suggest_float("threshold", 0.6, 0.99)
    k_clusters = trial.suggest_int("k_clusters", 2, 20)

    # 確保模型評估使用 VAL_CSV
    class_cmd = ["python", "svm.py", TRAIN_CSV, VAL_CSV, kernel, str(threshold)]
    cluster_cmd = ["main.exe", SVM_PRED_CSV, VAL_CSV, str(k_clusters)]
    
    return execute_pipeline(class_cmd, cluster_cmd, SVM_DIR, KMEANS_DIR)

# 主執行區塊
if __name__ == "__main__":
    print("開始執行 Optuna 驗證集自動調參...")
    
    objective_list = [objective_knn_kmeans,
                      objective_svm_kmeans,
                      objective_knn_dbscan,
                      objective_svm_dbscan]

    for target_objective in objective_list:

        combo_name = target_objective.__name__
        print("-" * 40)
        print(f"目前測試組合: {combo_name}")
        print("-" * 40)

        # 設定 TPESampler 與 random seed 以確保實驗具備可重現性
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(target_objective, n_trials=50)

        print(f"\n[{combo_name}] 調參結束")
        print("最佳參數組合:", study.best_params)
        print(f"驗證集最高準確率: {study.best_value:.4f}")
        print("\n" + "-" * 40 + "\n")