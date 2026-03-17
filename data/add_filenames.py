import pandas as pd
import os

# paths
csv_path = r"/data/au_data_jaffe/jaffe_cleaned.csv"
img_folder = r"C:\Users\fifia\PycharmProjects\facs_project\data\au_data_jaffe\jaffe_jpg_original"


# load the AU data
df = pd.read_csv(csv_path)

# get sorted list of filenames
filenames = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])

# sanity check: number of rows must match number of images
print("Rows in CSV:", len(df))
print("Images:", len(filenames))

if len(df) != len(filenames):
    raise ValueError("row count does not match number of images")

# add filename column
df["filename"] = filenames

# save updated CSV
df.to_csv(csv_path, index=False)

print("filenames added")
