import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Test Confusion Matrix (Your Final Results)
# ===============================
cm = np.array([
    [3004,    5,   14,    8,   13,   17],
    [   9,  205,   36,   25,   27,   34],
    [  43,  177,  380,    6,    8,  248],
    [   0,    4,    0,   92,   51,    5],
    [  31,    6,   13,   95,  469,   49],
    [  15,  184,  142,   18,   76, 2996]
])

# Class index → name mapping
classes = [
    "0: Normal",
    "1: PVC",
    "2: PAC",
    "3: VT",
    "4: SVT",
    "5: AFib"
]

# ===============================
# RAW CONFUSION MATRIX HEATMAP
# ===============================
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.title("Test Confusion Matrix (Raw Counts)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_raw.png", dpi=200)
plt.close()

# ===============================
# NORMALIZED HEATMAP
# ===============================
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.title("Test Confusion Matrix (Normalized)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=200)
plt.close()

print("Saved confusion_matrix_raw.png and confusion_matrix_normalized.png")
