import os
import numpy as np
import pandas as pd
from keras.models import load_model 
from keras.preprocessing.image import load_img, img_to_array 
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split

# Paths (adjust as necessary)
PREPROCESSED_DIR = r'D:\squid game\val\preprocessed'
LABELS_CSV = os.path.join(PREPROCESSED_DIR, 'labels.csv')
MODEL_SAVE_PATH = 'image_quality_analyzer_model.h5'

# Load the pre-trained model (from Day 3)
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense 
from tensorflow.keras.optimizers import Adam 
input_shape = (224, 224, 3)

# Rebuild the model architecture (same as Day 3)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load label data
labels_df = pd.read_csv(LABELS_CSV)

# Map labels and issues to categorical classes: Good=0, Bad-Blur=1, Bad-LowContrast=2
def map_class(row):
    if row['label'] == 'Good':
        return 0
    elif row['label'] == 'Bad':
        if 'blur' in row['issue'].lower():
            return 1
        elif 'lowcontrast' in row['issue'].replace(" ", "").lower():
            return 2
        else:
            return 1  # default Bad class
    else:
        return 0  # fallback Good

labels_df['class'] = labels_df.apply(map_class, axis=1)

# Load images and labels into arrays
X = []
y = []

print("Loading images for training...")

for idx, row in labels_df.iterrows():
    img_path = os.path.join(PREPROCESSED_DIR, row['filename'])
    img = load_img(img_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0  # Normalize
    X.append(img_arr)
    y.append(row['class'])

X = np.array(X, dtype='float32')
y = to_categorical(y, num_classes=3)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val, y_val))

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model trained and saved at: {MODEL_SAVE_PATH}")
