import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

# Directory containing preprocessed images and labels.csv (same as Day-4)
PREPROCESSED_DIR = r'D:\squid game\val\preprocessed'

# CSV file with labels and issues
LABELS_CSV = os.path.join(PREPROCESSED_DIR, 'labels.csv')

# Path to saved model from Day-4
MODEL_SAVE_PATH = 'image_quality_analyzer_model.h5'

# Load the trained model
model = load_model(MODEL_SAVE_PATH)

# Load label data
labels_df = pd.read_csv(LABELS_CSV)

# Map label and issue into class (same as Day-4)
def map_class(row):
    if row['label'] == 'Good':
        return 0
    elif row['label'] == 'Bad':
        if 'blur' in row['issue'].lower():
            return 1
        elif 'lowcontrast' in row['issue'].replace(" ", "").lower():
            return 2
        else:
            return 1  # default Bad
    else:
        return 0  # fallback Good

labels_df['class'] = labels_df.apply(map_class, axis=1)

# Load test images and true classes
X_test = []
y_test = []
print("Loading images for evaluation...")
for idx, row in labels_df.iterrows():
    img_path = os.path.join(PREPROCESSED_DIR, row['filename'])
    img = load_img(img_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0  # normalize to [0,1]
    X_test.append(img_arr)
    y_test.append(row['class'])

X_test = np.array(X_test, dtype='float32')
y_test = np.array(y_test)

# Predict class probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to predicted class indices
y_pred = np.argmax(y_pred_prob, axis=1)

# Define all class labels and names
labels = [0, 1, 2]
target_names = ['Good', 'Bad-Blur', 'Bad-LowContrast']

# Generate and print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))

# Generate and print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
