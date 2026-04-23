import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

merged_csv = r"C:\Users\fifia\PycharmProjects\facs_project\rafdb_aus_merged.csv"

model_path = r"C:\Users\fifia\PycharmProjects\facs_project\mlp model\mlp_emotion_model.pkl"
scaler_path = r"C:\Users\fifia\PycharmProjects\facs_project\mlp model\scaler.pkl"
label_encoder_path = r"C:\Users\fifia\PycharmProjects\facs_project\mlp model\label_encoder.pkl"


df = pd.read_csv(merged_csv)

df.columns = df.columns.str.strip()

df = df.rename(columns={col: f" {col}" for col in df.columns if col.startswith("AU")})

print(df.columns.tolist())  # confirm AU columns now have leading spaces

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# Select AU intensity columns (WITH leading space)
au_cols = [f" {c}" for c in [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r",
    "AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r",
    "AU20_r","AU23_r","AU25_r","AU26_r","AU45_r"
]]

X = df[au_cols]
y_true = df["emotion"]

# Encode true labels using JAFFE label encoder
y_true_encoded = label_encoder.transform(y_true)


# Scale AU features
X_scaled = scaler.transform(X)


# Predict using JAFFE model
y_pred = model.predict(X_scaled)

# Evaluate
print("RAF-DB Accuracy:", accuracy_score(y_true_encoded, y_pred))
print("\nClassification Report:\n", classification_report(y_true_encoded, y_pred))


# Save predictions
df["predicted"] = y_pred
df.to_csv(r"C:\Users\fifia\PycharmProjects\facs_project\rafdb_predictions.csv", index=False)

print("Predictions saved to rafdb_predictions.csv")
