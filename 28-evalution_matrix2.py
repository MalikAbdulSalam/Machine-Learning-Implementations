# ==========================================
# CLASSIFICATION EVALUATION
# ==========================================

# Import libraries

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ==========================================
# 1. CREATE DATASET
# ==========================================

# Features:
# Age and Salary

X = np.array([
    [20, 20000],
    [22, 25000],
    [25, 30000],
    [27, 35000],
    [30, 40000],
    [32, 45000],
    [35, 50000],
    [38, 55000],
    [40, 60000],
    [42, 65000],
    [45, 70000],
    [48, 75000],
    [50, 80000],
    [52, 85000],
    [55, 90000],
    [58, 95000],
    [60, 100000],
    [62, 105000],
    [65, 110000],
    [68, 115000]
])


# Target:
# 0 = No
# 1 = Yes

y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])


# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. CREATE MODEL
# ==========================================

model = LogisticRegression()


# ==========================================
# 4. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 5. MAKE CLASS PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. GET PREDICTED PROBABILITIES
# ==========================================

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print(cm)


# ==========================================
# 8. GET TN, FP, FN, TP
# ==========================================

TN, FP, FN, TP = cm.ravel()

print("\nTrue Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)


# ==========================================
# 9. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)


# ==========================================
# 10. PRECISION
# ==========================================

precision = precision_score(
    y_test,
    y_pred
)

print("Precision:", precision)


# ==========================================
# 11. RECALL
# ==========================================

recall = recall_score(
    y_test,
    y_pred
)

print("Recall:", recall)


# ==========================================
# 12. F1-SCORE
# ==========================================

f1 = f1_score(
    y_test,
    y_pred
)

print("F1 Score:", f1)


# ==========================================
# 13. SPECIFICITY
# ==========================================

specificity = TN / (TN + FP)

print("Specificity:", specificity)


# ==========================================
# 14. ROC-AUC
# ==========================================

auc = roc_auc_score(
    y_test,
    y_probability
)

print("ROC-AUC:", auc)


# ==========================================
# 15. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 16. ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


plt.plot(
    fpr,
    tpr,
    label="ROC Curve"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Model"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()