# ---------------------------------
# Imports
# ---------------------------------
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------------
# Load and Preprocess Dataset
# ---------------------------------

# Load dental X-ray dataset from .npz file
npdata = np.load('DENTAL1.NPZ')

# Extract train/test sets
x_train, y_train = npdata['x_train'], npdata['y_train']
x_test, y_test = npdata['x_test'], npdata['y_test']

# Ensure data is in (samples, 256, 256, 3) format
if x_train.ndim == 3:  # In case channels are missing
    x_train = x_train.reshape((-1, 256, 256, 3))
    x_test = x_test.reshape((-1, 256, 256, 3))

# Display dataset shapes
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape:  {x_test.shape}")
print(f"y_test shape:  {y_test.shape}")

# ---------------------------------
# Add Noise to Data
# ---------------------------------

# Set noise level
noise_factor = 0.2

# Add Gaussian noise and clip pixel values to [0, 1]
x_train_noisy = np.clip(x_train + noise_factor * tf.random.normal(shape=x_train.shape), 0., 1.)
x_test_noisy = np.clip(x_test + noise_factor * tf.random.normal(shape=x_test.shape), 0., 1.)

# ---------------------------------
# Visualize Original vs Noisy Images
# ---------------------------------

def show_noisy_images(x_clean, x_noisy, num_images=5):
    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        # Original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(x_clean[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Original', fontsize=12)
        
        # Noisy image
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(x_noisy[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Noisy', fontsize=12)

    plt.tight_layout()
    plt.show()

show_noisy_images(x_train, x_train_noisy)

# ---------------------------------
# Define the Autoencoder Model
# ---------------------------------

class DenoiseAutoencoder(keras.Model):
    def __init__(self):
        super(DenoiseAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = Sequential([
            layers.Input(shape=(256, 256, 3)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)
        ])
        
        # Decoder
        self.decoder = Sequential([
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
    
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate and compile the model
autoencoder = DenoiseAutoencoder()
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ---------------------------------
# Train the Autoencoder
# ---------------------------------

history = autoencoder.fit(
    x_train_noisy, x_train,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)

# ---------------------------------
# Plot Training History
# ---------------------------------

def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training vs Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# ---------------------------------
# Evaluate Model
# ---------------------------------

test_loss, test_mae = autoencoder.evaluate(x_test_noisy, x_test, verbose=0)
print(f"Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

# ---------------------------------
# Visualize Denoised Results
# ---------------------------------

def show_denoised_results(x_noisy, x_denoised, num_images=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Noisy input
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(x_noisy[i], cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        # Denoised output
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(x_denoised[i], cmap='gray')
        plt.title("Denoised")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Predict denoised images
denoised_images = autoencoder.predict(x_test_noisy)

# Show the results
show_denoised_results(x_test_noisy, denoised_images)
