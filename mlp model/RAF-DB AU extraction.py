import os
import subprocess

# path to RAF-DB test folder
rafdb_path = r"C:\Users\fifia\PycharmProjects\facs_project\test"

# output directory for OpenFace CSV files
output_dir = r"C:\Users\fifia\PycharmProjects\facs_project\rafdb_openface"
os.makedirs(output_dir, exist_ok=True)

# path to OpenFace FeatureExtraction binary
openface_bin = r"C:\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

# valid image extensions
valid_ext = {".jpg", ".jpeg", ".png"}

for root, dirs, files in os.walk(rafdb_path):
    for file in files:
        if os.path.splitext(file)[1].lower() in valid_ext:

            img_path = os.path.join(root, file)

            # run OpenFace FeatureExtraction
            cmd = [
                openface_bin,
                "-f", img_path,
                "-out_dir", output_dir,
                "-aus",
                "-2Dfp",
                "-3Dfp",
                "-pose",
                "-gaze"
            ]

            print("processing:", img_path)
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("AUs extracted")