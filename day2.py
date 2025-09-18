import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np

# Update these paths as per your setup
RAW_DATA_DIR = r'D:\squid game\test\test'        # Your raw dataset folder
PREPROCESSED_DIR = r'D:\squid game\test\test\preprocessed'  # Output folder for processed images and CSV

os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def is_blurry(img_gray, threshold=100):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def is_low_contrast(img_gray, threshold=50):
    return img_gray.std() < threshold

records = []

print("Starting Day 2: Data Preprocessing...")

def read_image(path):
    # Try OpenCV first
    img = cv2.imread(path)
    if img is not None:
        return img
    # Fallback to PIL if OpenCV fails (handles some corrupted images)
    try:
        pil_img = Image.open(path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"Image read failed: {path} ({e})")
        return None

# Recursively scan folders for images
for root, dirs, files in os.walk(RAW_DATA_DIR):
    for filename in files:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        file_path = os.path.join(root, filename)
        img = read_image(file_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # Detect blur or low contrast
        if is_blurry(gray):
            detected_issue = 'Blur'
        elif is_low_contrast(gray):
            detected_issue = 'LowContrast'
        else:
            detected_issue = 'None'
        # Infer label from folder name (if contains "good" or "bad")
        path_lower = root.lower()
        if 'good' in path_lower:
            label = 'Good'
        elif 'bad' in path_lower:
            label = 'Bad'
        else:
            label = 'Unknown'
        # Save image to PREPROCESSED_DIR (flattened)
        output_path = os.path.join(PREPROCESSED_DIR, filename)
        cv2.imwrite(output_path, img_resized)
        # Record info for CSV
        records.append([filename, label, detected_issue])

# Save CSV labels file
labels_df = pd.DataFrame(records, columns=['filename', 'label', 'issue'])
csv_path = os.path.join(PREPROCESSED_DIR, 'labels.csv')
labels_df.to_csv(csv_path, index=False)

print(f"Preprocessing complete. Processed images saved to: {PREPROCESSED_DIR}")
print(f"Labels CSV saved at: {csv_path}")
print("Day 2 completed successfully.")