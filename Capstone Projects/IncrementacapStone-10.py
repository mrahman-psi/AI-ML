### ================= TASK A: EfficientNetB0 Model =================

# --- Imports ---
import os
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Set Directories ---
output_dir = 'processed_data'
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# --- Image Parameters ---
img_height, img_width = 128, 128
batch_size = 1
testepochs = 6

# --- Check GPU Availability ---
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU")
else:
    print("Using CPU")

# --- Organize Images into Train/Test Split ---
def organize_images(folder_groups, split_ratio=0.8):
    '''
    Organizes images into training and testing directories.
    '''
    # train_dir = os.path.join(output_dir, 'train')
    # test_dir = os.path.join(output_dir, 'test')

    for class_name, _ in folder_groups:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for class_name, source_folders in folder_groups:
        all_files = []
        for folder in source_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                files = list(folder_path.glob('*'))
                all_files.extend([f for f in files if f.is_file()])

        random.shuffle(all_files)
        split_idx = int(split_ratio * len(all_files))

        for f in all_files[:split_idx]:
            shutil.copy(f, os.path.join(train_dir, class_name, f.name))
        for f in all_files[split_idx:]:
            shutil.copy(f, os.path.join(test_dir, class_name, f.name))

# --- Prepare Dataset ---
base_dir = os.getcwd()
folder_groups = [
    ('with', [os.path.join(base_dir, 'with_mask_pt1'), os.path.join(base_dir, 'with_mask_pt2')]),
    ('without', [os.path.join(base_dir, 'without_mask')])
]
organize_images(folder_groups, split_ratio=0.8)

# --- Data Generators ---
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# --- Build EfficientNetB0 Model ---
effnet_base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
effnet_base.trainable = False

model_effnet = Sequential([
    effnet_base,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model_effnet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# --- Train EfficientNetB0 Model ---
history_effnet = model_effnet.fit(
    train_data,
    validation_data=val_data,
    epochs=testepochs,
    callbacks=callbacks
)

# --- Plot History ---
def plot_history(history, title=''):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy ' + title)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss ' + title)
    plt.legend()
    plt.grid(True)

    plt.show()

plot_history(history_effnet, title='(EfficientNetB0)')

### ================= TASK B: ResNet50 Model =================

# --- Build ResNet50 Model ---
resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
resnet_base.trainable = False

model_resnet = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model_resnet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train ResNet50 Model ---
history_resnet = model_resnet.fit(
    train_data,
    validation_data=val_data,
    epochs=testepochs,
    callbacks=callbacks
)

plot_history(history_resnet, title='(ResNet50)')

# --- Predict and Visualize on Test Set ---
preds_resnet = model_resnet.predict(test_data)
predicted_classes_resnet = np.argmax(preds_resnet, axis=1)
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

plt.figure(figsize=(20, 10))
for i in range(10):
    img_path = os.path.join(test_data.directory, test_data.filenames[i])
    img = load_img(img_path, target_size=(img_height, img_width))
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"True: {class_labels[true_classes[i]]}\nPred: {class_labels[predicted_classes_resnet[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

### ================= TASK C: Compare and Best Model =================

# --- Evaluate Models ---
loss_effnet, acc_effnet = model_effnet.evaluate(test_data, verbose=0)
loss_resnet, acc_resnet = model_resnet.evaluate(test_data, verbose=0)

print(f"EfficientNetB0 Test Accuracy: {acc_effnet:.4f}")
print(f"ResNet50 Test Accuracy: {acc_resnet:.4f}")

# --- Choose Best Model ---
best_model = model_effnet if acc_effnet > acc_resnet else model_resnet
best_model_name = 'EfficientNetB0' if acc_effnet > acc_resnet else 'ResNet50'
print(f"Best Model: {best_model_name}")

# --- Predict and Plot with Best Model ---
preds_best = best_model.predict(test_data)
predicted_classes_best = np.argmax(preds_best, axis=1)

plt.figure(figsize=(20, 10))
for i in range(10):
    img_path = os.path.join(test_data.directory, test_data.filenames[i])
    img = load_img(img_path, target_size=(img_height, img_width))
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"True: {class_labels[true_classes[i]]}\nPred: {class_labels[predicted_classes_best[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
