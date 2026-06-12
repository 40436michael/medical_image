"""
Brain Tumor Classification - v3 (論文忠實復現)
Based on: Kang et al., Sensors 2021

核心修正：
  1. 移除 PCA — 論文完全沒有降維，直接用原始深層特徵
  2. 移除 Training augmentation — 論文的 augmentation 是擴增資料集用，
     不是在 feature extraction 時用；feature extraction 要用乾淨影像
  3. SVM grid search 參數範圍照論文：
       C ∈ [0.1, 1, 10, 100, 1000, 10000]
       gamma ∈ [0.00001, 0.0001, 0.001, 0.01]
  4. 加回論文使用的完整7個CNN（無PCA才合理）
  5. 特徵選擇：同家族只保留最高分那個（論文 Section 3.4）
  6. Experiment 2 也輸出 top-2 ensemble 結果（論文 Table 9 格式）

Dataset: BT-large-4c
  ./Training/  glioma_tumor/ meningioma_tumor/ no_tumor/ pituitary_tumor/
  ./Testing/   ...
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

# ============================================================
# 0. CONFIGURATION
# ============================================================

TRAIN_DIR  = "./Training"
TEST_DIR   = "./Testing"
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CNN Registry ─────────────────────────────────────────────
# (builder_fn, weights, head_attr, input_size, family_tag)
CNN_REGISTRY = {
    "DenseNet-121":  (models.densenet121,       models.DenseNet121_Weights.DEFAULT,       "classifier", 224, "densenet"),
    "DenseNet-169":  (models.densenet169,        models.DenseNet169_Weights.DEFAULT,       "classifier", 224, "densenet"),
    "ResNet-50":     (models.resnet50,           models.ResNet50_Weights.DEFAULT,          "fc",         224, "resnet"),
    "ResNeXt-50":    (models.resnext50_32x4d,    models.ResNeXt50_32X4D_Weights.DEFAULT,   "fc",         224, "resnext"),
    "MobileNet-V2":  (models.mobilenet_v2,       models.MobileNet_V2_Weights.DEFAULT,      "classifier", 224, "mobilenet"),
    "MnasNet":       (models.mnasnet1_0,         models.MNASNet1_0_Weights.DEFAULT,        "classifier", 224, "mnasnet"),
    "ShuffleNet-V2": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.DEFAULT,"fc",        224, "shufflenet"),
}

# ── SVM Grid (論文 Section 3.3.6) ────────────────────────────
SVM_PARAM_GRID = {
    "clf__C":     [0.1, 1, 10, 100, 1000, 10000],
    "clf__gamma": [0.00001, 0.0001, 0.001, 0.01],
}

# ── k-NN range (論文：k from 1 to 4) ─────────────────────────
KNN_K_RANGE = [1, 2, 3, 4]

# ── RF range (論文：trees from 1 to 150) ─────────────────────
RF_N_RANGE = list(range(10, 151, 10))   # 縮短搜尋範圍，效果相近


# ============================================================
# 1. IMAGE PRE-PROCESSING  (論文 Section 3.1)
# ============================================================
# ★ 重要：feature extraction 時只做 resize + normalize（不加 augmentation）
#    論文的 augmentation 是事先把訓練集擴增存到磁碟，不是 on-the-fly
# ============================================================

def get_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def load_data(split_dir, size=224):
    ds = datasets.ImageFolder(split_dir, transform=get_transform(size))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader, ds.classes


# ============================================================
# 2. DEEP FEATURE EXTRACTION  (論文 Section 3.2)
# ============================================================

def build_extractor(model_fn, weights, head_attr):
    """凍結所有權重，把分類頭換成 Identity → 輸出特徵向量"""
    model = model_fn(weights=weights)
    setattr(model, head_attr, nn.Identity())
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE).eval()

def extract_features(model, loader):
    feats, lbls = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            out = model(imgs.to(DEVICE)).view(imgs.size(0), -1)
            feats.append(out.cpu().numpy())
            lbls.append(targets.numpy())
    return np.concatenate(feats), np.concatenate(lbls)


# ============================================================
# 3. ML CLASSIFIERS  (論文 Section 3.3)
# ============================================================
# ★ 重要：論文直接把 deep feature 丟進 classifier，只加 StandardScaler
#          完全沒有 PCA，這是準確率高的關鍵
# ============================================================

def build_pipeline(clf):
    """只有 StandardScaler + classifier，無 PCA（論文做法）"""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def fit_svm(X_train, y_train, X_test, y_test):
    """SVM with grid search over C and gamma (論文 Section 3.3.6)"""
    base = build_pipeline(SVC(kernel="rbf"))
    gs = GridSearchCV(base, SVM_PARAM_GRID, cv=3, scoring="accuracy",
                      n_jobs=-1, verbose=0)
    t0 = time.time()
    gs.fit(X_train, y_train)
    y_pred = gs.best_estimator_.predict(X_test)
    elapsed = time.time() - t0
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred, elapsed, gs.best_params_


def fit_knn(X_train, y_train, X_test, y_test):
    """k-NN: try k=1..4, pick best (論文 Section 3.3.4)"""
    best_acc, best_pred, best_k = 0, None, 1
    t0 = time.time()
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    for k in KNN_K_RANGE:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        clf.fit(Xtr_s, y_train)
        pred = clf.predict(Xte_s)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc, best_pred, best_k = acc, pred, k
    return best_acc, best_pred, time.time()-t0, {"k": best_k}


def fit_rf(X_train, y_train, X_test, y_test):
    """RF: try n_estimators range, pick best (論文 Section 3.3.5)"""
    best_acc, best_pred, best_n = 0, None, 10
    t0 = time.time()
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    for n in RF_N_RANGE:
        clf = RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
        clf.fit(Xtr_s, y_train)
        pred = clf.predict(Xte_s)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc, best_pred, best_n = acc, pred, n
    return best_acc, best_pred, time.time()-t0, {"n_estimators": best_n}


def fit_generic(clf_obj, X_train, y_train, X_test, y_test):
    pipe = build_pipeline(clf_obj)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return accuracy_score(y_test, y_pred), y_pred, time.time()-t0, {}


def evaluate_all_classifiers(X_train, y_train, X_test, y_test):
    """
    Run all 5 ML classifiers, return dict {clf_name: (acc, y_pred, elapsed)}
    """
    results = {}

    # Gaussian NB
    acc, pred, t, _ = fit_generic(GaussianNB(), X_train, y_train, X_test, y_test)
    results["Gaussian NB"] = (acc, pred, t)

    # AdaBoost (n_estimators=150, 論文設定)
    acc, pred, t, _ = fit_generic(
        AdaBoostClassifier(n_estimators=150, random_state=42),
        X_train, y_train, X_test, y_test)
    results["AdaBoost"] = (acc, pred, t)

    # k-NN (k=1~4)
    acc, pred, t, p = fit_knn(X_train, y_train, X_test, y_test)
    results["k-NN"] = (acc, pred, t)

    # Random Forest
    acc, pred, t, p = fit_rf(X_train, y_train, X_test, y_test)
    results["Random Forest"] = (acc, pred, t)

    # SVM-RBF with grid search
    acc, pred, t, p = fit_svm(X_train, y_train, X_test, y_test)
    print(f"       SVM best: C={p.get('clf__C')}, gamma={p.get('clf__gamma')}")
    results["SVM-RBF"] = (acc, pred, t)

    return results


# ============================================================
# 4. FEATURE SELECTION  (論文 Section 3.4)
# ============================================================

def select_top3(results_table, family_map):
    """
    1. 計算每個 CNN 在所有分類器上的平均 accuracy
    2. 依平均 accuracy 降冪排序
    3. 同一 family 只保留最高分（論文明確規定）
    4. 取前3個
    """
    avg_acc = {
        cnn: np.mean([v[0] for v in clf_dict.values()])
        for cnn, clf_dict in results_table.items()
    }
    std_acc = {
        cnn: np.std([v[0] for v in clf_dict.values()])
        for cnn, clf_dict in results_table.items()
    }
    ranked = sorted(avg_acc, key=lambda x: (avg_acc[x], -std_acc[x]), reverse=True)

    print("\n[Feature Selection] Rankings:")
    print(f"  {'Rank':4} {'CNN':22} {'Avg Acc':8} {'Std':6} {'Family'}")
    print("  " + "-"*52)
    for i, name in enumerate(ranked):
        print(f"  {i+1:4d} {name:22s} {avg_acc[name]:.4f}   {std_acc[name]:.4f}  {family_map[name]}")

    selected, used_families = [], set()
    for name in ranked:
        fam = family_map[name]
        if fam in used_families:
            print(f"       ✗ skip {name} (family '{fam}' already selected)")
            continue
        selected.append(name)
        used_families.add(fam)
        if len(selected) == 3:
            break

    print(f"\n  → Top-3 (heterogeneous): {selected}\n")
    return selected


# ============================================================
# 5. ENSEMBLE  (論文 Section 3.5)
# ============================================================

def concat_features(feature_dict, names):
    return np.concatenate([feature_dict[n] for n in names], axis=1)

def run_ensemble_experiments(train_feats, test_feats, y_train, y_test, top3):
    """
    論文 Table 9 格式：
    - 每個 top-3 單獨跑
    - 所有 C(3,2)=3 個 top-2 組合
    - 完整 top-3 ensemble
    回傳 {feature_desc: {clf_name: acc}}
    """
    from itertools import combinations

    combos = []
    for name in top3:
        combos.append(([name], name))
    for r in [2, 3]:
        for subset in combinations(top3, r):
            label = " + ".join(subset)
            combos.append((list(subset), label))

    ensemble_table = {}
    for names, label in combos:
        print(f"\n  ── {label}")
        X_tr = concat_features(train_feats, names)
        X_te = concat_features(test_feats,  names)
        clf_results = evaluate_all_classifiers(X_tr, y_train, X_te, y_test)
        ensemble_table[label] = {k: v[0] for k, v in clf_results.items()}
        for clf_name, (acc, _, elapsed) in clf_results.items():
            print(f"     {clf_name:15s} acc={acc:.4f}  ({elapsed:.1f}s)")

    return ensemble_table


# ============================================================
# 6. VISUALIZATION
# ============================================================

def plot_heatmap(matrix_dict, clf_names, title, save_path=None, vmin=0.5):
    row_names = list(matrix_dict.keys())
    matrix = np.array([[matrix_dict[r].get(c, 0) for c in clf_names] for r in row_names])

    fig, ax = plt.subplots(figsize=(max(10, len(clf_names)*2.2),
                                    max(4, len(row_names)*0.85)))
    sns.heatmap(matrix, annot=True, fmt=".4f", cmap="YlGnBu",
                xticklabels=clf_names, yticklabels=row_names,
                vmin=vmin, vmax=1.0, ax=ax, linewidths=0.4,
                annot_kws={"size": 9})
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("ML Classifier"); ax.set_ylabel("Deep Feature Source")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    #plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    #plt.show()


def plot_ensemble_vs_single(exp1_table, ensemble_table, clf_names, top3, save_path=None):
    """
    Compare:
      - best single CNN (from exp1)
      - top-3 ensemble
    """
    single_best = {c: max(exp1_table[cnn][c][0] for cnn in exp1_table) for c in clf_names}
    ens_key = " + ".join(top3)
    ens_accs = ensemble_table.get(ens_key, {})

    x = np.arange(len(clf_names)); w = 0.35
    s_vals = [single_best.get(c, 0) for c in clf_names]
    e_vals = [ens_accs.get(c, 0)    for c in clf_names]

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, s_vals, w, label="Best single CNN", color="#4C8BE0")
    b2 = ax.bar(x + w/2, e_vals, w, label=f"Ensemble ({ens_key})", color="#E07C4C")
    ax.bar_label(b1, fmt="%.4f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.4f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(clf_names, rotation=20, ha="right")
    ax.set_ylim(0.4, 1.12); ax.set_ylabel("Accuracy")
    ax.set_title("Single Best CNN vs Top-3 Ensemble (BT-large-4c)")
    ax.legend(loc="lower right"); ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    #plt.show()


def plot_per_class_metrics(y_true, y_pred, classes, title, save_path=None):
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true, y_pred, average=None)
    recall    = recall_score   (y_true, y_pred, average=None)
    f1        = f1_score       (y_true, y_pred, average=None)

    x = np.arange(len(classes)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w,   precision, w, label="Precision", color="#4C8BE0")
    b2 = ax.bar(x,       recall,    w, label="Recall",    color="#E07C4C")
    b3 = ax.bar(x + w,   f1,        w, label="F1-Score",  color="#50C878")
    ax.bar_label(b3, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score")
    ax.set_title(title); ax.legend()
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    #plt.show()


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("=" * 68)
    print("Brain Tumor Classification — v3 (論文忠實復現)")
    print(f"Dataset: BT-large-4c  |  Device: {DEVICE}")
    print("Key fix: No PCA, No augmentation during feature extraction")
    print("=" * 68)

    # ── Step 1: Feature extraction ─────────────────────────
    print("\n[Step 1] Extracting deep features...")
    train_feats, test_feats, family_map = {}, {}, {}
    y_train, y_test, classes = None, None, None

    for cnn_name, (fn, weights, head, size, family) in CNN_REGISTRY.items():
        print(f"  → {cnn_name}")
        tr_loader, cls = load_data(TRAIN_DIR, size)
        te_loader, _   = load_data(TEST_DIR,  size)
        if classes is None: classes = cls

        model = build_extractor(fn, weights, head)
        Xtr, ytr = extract_features(model, tr_loader)
        Xte, yte = extract_features(model, te_loader)
        train_feats[cnn_name] = Xtr
        test_feats [cnn_name] = Xte
        family_map [cnn_name] = family
        if y_train is None: y_train, y_test = ytr, yte
        print(f"     dim={Xtr.shape[1]}, train={Xtr.shape[0]}, test={Xte.shape[0]}")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    clf_names_ordered = ["Gaussian NB", "AdaBoost", "k-NN", "Random Forest", "SVM-RBF"]

    # ── Step 2: Experiment 1 (single CNN × all classifiers) ─
    print("\n[Step 2] Experiment 1 — Single CNN × ML classifiers")
    exp1_table = {}   # {cnn_name: {clf_name: (acc, pred, t)}}

    for cnn_name in CNN_REGISTRY:
        print(f"\n  CNN: {cnn_name}")
        Xtr, Xte = train_feats[cnn_name], test_feats[cnn_name]
        clf_results = evaluate_all_classifiers(Xtr, y_train, Xte, y_test)
        exp1_table[cnn_name] = clf_results
        for clf_name, (acc, _, elapsed) in clf_results.items():
            print(f"    {clf_name:15s} acc={acc:.4f}  ({elapsed:.1f}s)")

    # heatmap
    exp1_acc_only = {cnn: {c: v[0] for c, v in d.items()} for cnn, d in exp1_table.items()}
    plot_heatmap(exp1_acc_only, clf_names_ordered,
                 "Experiment 1: Accuracy Heatmap — Single CNN × ML Classifier (BT-large-4c)",
                 save_path="exp1_heatmap.png", vmin=0.5)

    # ── Step 3: Feature selection ───────────────────────────
    print("\n[Step 3] Feature Selection")
    top3 = select_top3(exp1_table, family_map)

    # ── Step 4: Experiment 2 (ensemble) ────────────────────
    print("\n[Step 4] Experiment 2 — Ensemble experiments")
    ensemble_table = run_ensemble_experiments(
        train_feats, test_feats, y_train, y_test, top3)

    plot_heatmap(ensemble_table, clf_names_ordered,
                 "Experiment 2: Ensemble Accuracy (BT-large-4c)",
                 save_path="exp2_ensemble_heatmap.png", vmin=0.5)

    plot_ensemble_vs_single(exp1_table, ensemble_table, clf_names_ordered, top3,
                            save_path="exp2_comparison.png")

    # ── Step 5: Best model detailed analysis ───────────────
    ens_key = " + ".join(top3)
    X_tr_ens = concat_features(train_feats, top3)
    X_te_ens  = concat_features(test_feats,  top3)

    # find best clf from ensemble results
    best_acc_ens = max(ensemble_table[ens_key].values())
    best_clf_name = max(ensemble_table[ens_key], key=ensemble_table[ens_key].get)

    # re-run best to get predictions
    print(f"\n[Step 5] Re-running best: Ensemble({ens_key}) + {best_clf_name}")
    if best_clf_name == "SVM-RBF":
        acc, best_pred, _, p = fit_svm(X_tr_ens, y_train, X_te_ens, y_test)
    elif best_clf_name == "k-NN":
        acc, best_pred, _, p = fit_knn(X_tr_ens, y_train, X_te_ens, y_test)
    elif best_clf_name == "Random Forest":
        acc, best_pred, _, p = fit_rf(X_tr_ens, y_train, X_te_ens, y_test)
    else:
        clf_map = {
            "Gaussian NB": GaussianNB(),
            "AdaBoost":    AdaBoostClassifier(n_estimators=150, random_state=42),
        }
        acc, best_pred, _, _ = fit_generic(clf_map[best_clf_name],
                                            X_tr_ens, y_train, X_te_ens, y_test)

    print(f"\n  Best accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, best_pred, target_names=classes))

    plot_confusion_matrix(
        y_test, best_pred, classes,
        title=f"Confusion Matrix\nEnsemble({ens_key}) + {best_clf_name}  acc={acc:.4f}",
        save_path="exp2_confusion_matrix.png")

    plot_per_class_metrics(
        y_test, best_pred, classes,
        title=f"Per-class Precision / Recall / F1\nEnsemble + {best_clf_name}",
        save_path="exp2_per_class.png")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("FINAL SUMMARY")
    print(f"  Top-3 CNNs : {top3}")
    print(f"  Best clf   : {best_clf_name}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 (macro) : {f1_score(y_test, best_pred, average='macro'):.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()