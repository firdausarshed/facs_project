import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import joblib
from matplotlib.colors import LinearSegmentedColormap

# colour palette
dark = "#501549"
mid = "#9A3C8A"
light = "#D68AD6"

# custom colormap for heatmaps
my_cmap = LinearSegmentedColormap.from_list(
    "custom_purple",
    [dark, mid, light]
)

# global colour cycle for bar plots etc
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[dark, mid, light])

# load saved data
y_test = joblib.load("y_test.pkl")
y_pred = joblib.load("y_pred.pkl")
le = joblib.load("label_encoder.pkl")
acc = joblib.load("accuracy.pkl")

classes = le.classes_

# confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=my_cmap,
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# precision, recall, f1 bar plot
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
x = np.arange(len(classes))

plt.figure(figsize=(10,6))
plt.bar(x - 0.2, prec, width=0.2, label="Precision", color=dark)
plt.bar(x, rec, width=0.2, label="Recall", color=mid)
plt.bar(x + 0.2, f1, width=0.2, label="F1 Score", color=light)

plt.xticks(x, classes, rotation=45)
plt.ylabel("Score")
plt.title("Precision, Recall, F1 per Emotion")
plt.legend()
plt.tight_layout()
plt.show()

# accuracy plot
plt.figure(figsize=(4,4))
plt.bar(["Accuracy"], [acc], color=mid)
plt.ylim(0,1)
plt.title("Model Accuracy")
plt.tight_layout()
plt.show()
