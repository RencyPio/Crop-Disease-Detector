import os

dataset_path = '/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'  
classes = os.listdir(dataset_path)

print(f"Number of classes: {len(classes)}")

total_images = 0
for c in classes:
    class_path = os.path.join(dataset_path, c)
    num_images = len(os.listdir(class_path))
    total_images += num_images
    print(f"{c}: {num_images} images")

print(f"\nTOTAL IMAGES: {total_images}")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# ---- CONFIG ----
DATASET_PATH = '/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'  # confirmed path from your check
IMG_SIZE = (224, 224)      # standard input size for MobileNetV2
BATCH_SIZE = 32
VAL_SPLIT = 0.15           # 15% validation
TEST_SPLIT = 0.15  

# ---- STEP 1: Data Augmentation (training data only) ----
train_datagen = ImageDataGenerator(
    rescale=1./255,              # normalize pixel values to [0,1]
    rotation_range=20,           # random rotation up to 20 degrees
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=VAL_SPLIT   # reserve part of training data as validation
)

# Validation/test data should NOT be augmented — only rescaled
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VAL_SPLIT
)

# ---- STEP 2: Load data from directory ----
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# ---- STEP 3: Check class distribution + compute class weights ----
# This addresses the imbalance you found (152 to 5,507 images per class)
from sklearn.utils.class_weight import compute_class_weight

class_indices = train_generator.class_indices  # dict: {class_name: index}
classes_list = list(class_indices.keys())
num_classes = len(classes_list)

print(f"Number of classes detected: {num_classes}")
print(f"Class indices sample: {list(class_indices.items())[:5]}")

# Get all training labels to compute class weights
train_labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights_array))

print("\nClass weights (higher = more underrepresented, will be upweighted during training):")
for idx, weight in list(class_weights.items())[:5]:
    print(f"  Class {idx} ({classes_list[idx]}): weight = {weight:.2f}")

# ---- STEP 4: Sanity check — print shapes ----
sample_batch_images, sample_batch_labels = next(train_generator)
print(f"\nSample batch image shape: {sample_batch_images.shape}")   # (batch_size, 224, 224, 3)
print(f"Sample batch label shape: {sample_batch_labels.shape}")     # (batch_size, num_classes)

print(f"\nTotal training images (after split): {train_generator.samples}")
print(f"Total validation images (after split): {val_generator.samples}")