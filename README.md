# Brain Tumor Classification Using Ensemble Deep Features and Machine Learning Classifiers

## 📖 Project Overview

本專案為醫療影像期末專題，目標為利用腦部 MRI（Magnetic Resonance Imaging）影像進行腦瘤分類（Brain Tumor Classification）。

本研究參考 Kang 等人於 Sensors 期刊發表之論文：

> MRI-Based Brain Tumor Classification Using Ensemble of Deep Features and Machine Learning Classifiers

並嘗試忠實復現（Reproduction）其方法，透過多個預訓練卷積神經網路（CNN）進行深層特徵擷取（Deep Feature Extraction），再搭配傳統機器學習分類器完成腦瘤分類任務。

---

## 🎯 Research Objective

利用 MRI 影像辨識四種類別：

1. Glioma Tumor
2. Meningioma Tumor
3. Pituitary Tumor
4. No Tumor

研究重點：

* 比較不同 CNN 特徵擷取器效能
* 比較不同 Machine Learning Classifiers 表現
* 實作論文提出之 Ensemble Deep Feature 方法
* 分析最佳模型之分類結果與混淆矩陣

---

## 📂 Dataset

資料集來源：

**Brain Tumor Classification (MRI)**

🔗 Kaggle Dataset：

https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

### Dataset Structure

```text
Training/
├── glioma_tumor/
├── meningioma_tumor/
├── pituitary_tumor/
└── no_tumor/

Testing/
├── glioma_tumor/
├── meningioma_tumor/
├── pituitary_tumor/
└── no_tumor/
```

### Classes

| Class      | Description |
| ---------- | ----------- |
| Glioma     | 神經膠質瘤       |
| Meningioma | 腦膜瘤         |
| Pituitary  | 腦下垂體腫瘤      |
| No Tumor   | 正常腦部 MRI    |

---

## 📑 Reference Paper

Kang, J., Ullah, Z., Gwak, J.

**MRI-Based Brain Tumor Classification Using Ensemble of Deep Features and Machine Learning Classifiers**

Sensors, 2021.

DOI:

https://doi.org/10.3390/s21062229

PubMed:

https://pubmed.ncbi.nlm.nih.gov/33810176/

---

## 🧠 Methodology

### 1. Image Preprocessing

依照論文設定：

* Resize → 224 × 224
* Normalize using ImageNet statistics

```python
Normalize(
 mean=[0.485, 0.456, 0.406],
 std =[0.229, 0.224, 0.225]
)
```

本專案在 Feature Extraction 階段：

✅ Resize

✅ Normalize

❌ PCA

❌ Online Data Augmentation

---

### 2. Deep Feature Extraction

使用 ImageNet 預訓練模型：

| CNN Model    |
| ------------ |
| DenseNet121  |
| DenseNet169  |
| ResNet50     |
| ResNeXt50    |
| MobileNetV2  |
| MnasNet      |
| ShuffleNetV2 |

分類頭（Classifier Head）移除後：

```python
nn.Identity()
```

取得深層特徵向量作為 Machine Learning 分類器輸入。

---

### 3. Machine Learning Classifiers

共比較五種分類器：

| Classifier           |
| -------------------- |
| Gaussian Naive Bayes |
| AdaBoost             |
| k-NN                 |
| Random Forest        |
| SVM (RBF Kernel)     |

---

### SVM Hyperparameter Search

依照論文設定：

```python
C = [0.1, 1, 10, 100, 1000, 10000]

gamma = [
0.00001,
0.0001,
0.001,
0.01
]
```

Grid Search：

```python
GridSearchCV()
```

---

### k-NN Search Range

```python
k = 1 ~ 4
```

---

### Random Forest Search Range

```python
n_estimators = 10 ~ 150
```

---

## 🔍 Feature Selection

根據論文 Section 3.4：

1. 計算每個 CNN 在所有分類器上的平均 Accuracy
2. 依 Accuracy 排序
3. 同家族 CNN 僅保留最高分模型
4. 選出 Top-3 CNN

範例：

```text
DenseNet169
ResNeXt50
MobileNetV2
```

---

## 🤝 Ensemble Learning

實驗包含：

### Single Feature

```text
DenseNet169
```

### Top-2 Ensemble

```text
DenseNet169 + ResNeXt50
DenseNet169 + MobileNetV2
ResNeXt50 + MobileNetV2
```

### Top-3 Ensemble

```text
DenseNet169
+ ResNeXt50
+ MobileNetV2
```

特徵融合方式：

```python
np.concatenate()
```

---

## 📊 Evaluation Metrics

使用以下指標：

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Scikit-Learn：

```python
classification_report()
confusion_matrix()
```

---

## 📈 Visualization

本專案輸出：

### Experiment 1

CNN × ML Classifier Accuracy Heatmap

```text
exp1_heatmap.png
```

### Experiment 2

Ensemble Accuracy Heatmap

```text
exp2_ensemble_heatmap.png
```

### Comparison

Single CNN vs Ensemble

```text
exp2_comparison.png
```

### Confusion Matrix

```text
exp2_confusion_matrix.png
```

### Per-Class Metrics

```text
exp2_per_class.png
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your_username/brain-tumor-classification.git

cd brain-tumor-classification
```

### Create Environment

```bash
conda create -n brain_tumor python=3.10

conda activate brain_tumor
```

### Install Dependencies

```bash
pip install torch torchvision

pip install numpy pandas

pip install matplotlib seaborn

pip install scikit-learn
```

---

## 🚀 Run

將資料集放置如下：

```text
project/

├── Training/
├── Testing/
├── main.py
```

執行：

```bash
python main.py
```

---

## 📋 Expected Output

```text
[Step 1] Feature Extraction

[Step 2] Experiment 1

[Step 3] Feature Selection

[Step 4] Ensemble Learning

[Step 5] Best Model Evaluation

FINAL SUMMARY
```

輸出：

```text
Accuracy
Macro F1
Confusion Matrix
Heatmaps
```

---

## 🏆 Project Contributions

* 忠實復現 Sensors 2021 論文方法
* 使用 7 種預訓練 CNN 提取深層特徵
* 比較 5 種機器學習分類器
* 實作 Top-3 Feature Ensemble
* 完成 MRI 腦瘤四分類任務
* 提供完整結果分析與視覺化

---

## 👨‍🎓 Course Information

Course: Medical Image Analysis

Project: Brain Tumor Classification Using MRI

Student: Your Name

Department: Department of Computer Science and Information Engineering

Semester: 2026 Spring

---

## 📜 License

This project is for academic and educational purposes only.

Dataset copyright belongs to the original authors and Kaggle dataset providers.
