import joblib

#load model
model = joblib.load("mlp_emotion_model.pkl")

#load scaler
scaler = joblib.load("scaler.pkl")

#load label encoder
le = joblib.load("model, scaler, label encoder/label_encoder.pkl")

