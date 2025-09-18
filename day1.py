import cv2
import os

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

def is_dull(image, threshold=0.07):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray.std() / 255.0
    return contrast < threshold

def classify_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Unreadable image"
    if is_blurry(image):
        return "Blurry"
    elif is_dull(image):
        return "Dull"
    else:
        return "Good"

def classify_dataset(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            label = classify_image(image_path)
            print(f"{filename}: {label}")

if __name__ == "__main__":
    dataset_folder = r"D:\squid game\val\val"  # Change to your dataset folder
    classify_dataset(dataset_folder)
   
   
