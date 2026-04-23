import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import joblib
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance

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
model = joblib.load("mlp_emotion_model.pkl")
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
feature_names = joblib.load("feature_names.pkl")

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

# learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, color=dark, label="Training Accuracy")
plt.plot(train_sizes, val_mean, color=light, label="Validation Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# loss Curve
plt.figure(figsize=(8,5))
plt.plot(model.loss_curve_, color=mid)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()

# permutation feature importance (AUs)
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

importances = result.importances_mean
indices = np.argsort(importances)

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices], color=dark)
plt.yticks(range(len(indices)), np.array(feature_names)[indices])
plt.xlabel("Importance Score")
plt.title("Permutation Feature Importance (AUs)")
plt.grid(axis='x')
plt.tight_layout()
plt.show()
