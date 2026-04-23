import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

csv_path = r"C:\Users\fifia\PycharmProjects\facs_project\rafdb_predictions.csv"
label_encoder_path = r"C:\Users\fifia\PycharmProjects\facs_project\mlp model\label_encoder.pkl"

df = pd.read_csv(csv_path)
label_encoder = joblib.load(label_encoder_path)

y_true = label_encoder.transform(df["emotion"])
y_pred = df["predicted"].astype(int)
classes = label_encoder.classes_


dark = "#501549"
mid = "#9A3C8A"
light = "#D68AD6"
palette = [dark, mid, light]

my_cmap = LinearSegmentedColormap.from_list(
    "custom_purple",
    [dark, mid, light]
)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=my_cmap,
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted", color=dark)
plt.ylabel("Actual", color=dark)
plt.title("Confusion Matrix (RAF-DB)", color=dark)
plt.tight_layout()
plt.show()

# Correlation scatter plot
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5, color=mid)
plt.xlabel("Actual (encoded)", color=dark)
plt.ylabel("Predicted (encoded)", color=dark)
plt.title("Correlation Between Actual and Predicted Labels", color=dark)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

corr = np.corrcoef(y_true, y_pred)[0,1]
print("Correlation:", corr)


# AU distribution boxplot
au_cols = [c for c in df.columns if c.endswith("_r")]

plt.figure(figsize=(12,6))
sns.boxplot(data=df[au_cols], palette=palette)
plt.xticks(rotation=90, color=dark)
plt.title("AU Intensity Distribution (RAF-DB)", color=dark)
plt.tight_layout()
plt.show()

# Accuracy + report
print("Overall Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))
