import os
from PIL import Image

input_folder = "./dataset/test/images_jpg"
output_folder = "./dataset/test/images"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".jpg"):
        img = Image.open(os.path.join(input_folder, file))
        # keep same filename but .png extension
        new_name = os.path.splitext(file)[0] + ".png"
        img.save(os.path.join(output_folder, new_name))
