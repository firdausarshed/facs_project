# code to add an emotion label column using the filenames (emotions indicated in filenames)
import pandas as pd

csv_path = r"C:\Users\fifia\PycharmProjects\facs_project\data\au_data_jaffe\jaffe_cleaned.csv"

df = pd.read_csv(csv_path)

def extract_emotion(fname):
    # jaffe filenames look like: KA.AN1.39.jpg
    # the emotion code is always the middle part, first two letters
    parts = fname.split(".")
    if len(parts) < 3:
        return "unknown"
    code = parts[1][:2]  # AN, DI, FE, HA, NE, SA, SU

    mapping = {
        "AN": "anger",
        "DI": "disgust",
        "FE": "fear",
        "HA": "happiness",
        "NE": "neutral",
        "SA": "sadness",
        "SU": "surprise"
    }
    return mapping.get(code, "unknown")

df["emotion"] = df["filename"].apply(extract_emotion)

df.to_csv(csv_path, index=False)

print("emotion labels added")
