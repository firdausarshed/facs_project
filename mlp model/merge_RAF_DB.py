import os
import pandas as pd

# folder where OpenFace saved the CSV files
csv_folder = r"C:\Users\fifia\PycharmProjects\facs_project\rafdb_openface"

# output merged file
output_csv = r"C:\Users\fifia\PycharmProjects\facs_project\rafdb_aus_merged.csv"

# RAF-DB folder (to read labels from folder names 1–7)
rafdb_root = r"C:\Users\fifia\PycharmProjects\facs_project\test"

all_rows = []

for root, dirs, files in os.walk(rafdb_root):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(root, file)

            # corresponding OpenFace CSV
            csv_name = file.replace(".jpg", ".csv").replace(".png", ".csv")
            csv_path = os.path.join(csv_folder, csv_name)

            if not os.path.exists(csv_path):
                continue  # skip missing CSVs

            # Load AU CSV
            df = pd.read_csv(csv_path)

            # Extract label from folder name (1–7)
            label = os.path.basename(root)

            df["label"] = int(label)
            df["filename"] = file

            all_rows.append(df)

# merge everything
merged = pd.concat(all_rows, ignore_index=True)

# save
merged.to_csv(output_csv, index=False)

print("Merged RAF-DB AU dataset saved to:", output_csv)
