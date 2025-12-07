import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix data
cm = np.array([
    [2994,    1,   19,    5,    8,   34],
    [  12,  182,   81,   34,    6,   21],
    [  47,  105,  500,    5,    5,  200],
    [   0,   16,    2,  113,   18,    3],
    [  23,   10,   23,  142,  407,   58],
    [  48,  183,  492,   20,   55, 2633]
])

classes = ["0", "1", "2", "3", "4", "5"]

# Normalized confusion matrix
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Normalized Confusion Matrix (PPG Arrhythmia Classification)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

out_path = "/mnt/data/confusion_matrix_normalized.png"
plt.savefig(out_path, dpi=200)
out_path
