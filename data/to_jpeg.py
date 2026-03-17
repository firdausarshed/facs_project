#converting pngs to jpeg to overcome openface file restrictions

from PIL import Image
import os

src = r"/data/datasets/jaffe_png"
dst = r"C:\Users\fifia\PycharmProjects\facs_project\data\jaffe_jpg"
os.makedirs(dst, exist_ok=True)

for f in os.listdir(src):
    if f.endswith('.png'):
        img = Image.open(os.path.join(src, f)).convert('RGB')
        img.save(os.path.join(dst, f.replace('.png', '.jpg')))