import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split

# DATA PREPROCESSING

# loading cleaned data
csv_path = r"C:\Users\fifia\PycharmProjects\facs_project\data\au_data_jaffe\jaffe_cleaned.csv"
df = pd.read_csv(csv_path)

# extracting AU features + emotion labels
X = df[[c for c in df.columns if c.endswith("_r")]]
y = df["emotion"]

# encoding emotion labels
le = prep.LabelEncoder()
y_encoded = le.fit_transform(y)

# scaling AU features
scaler = prep.StandardScaler()
X_scaled = scaler.fit_transform(X)

# train + test split (use X_scaled!)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Classes:", le.classes_)

# MODEL TRAINING

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluate
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
