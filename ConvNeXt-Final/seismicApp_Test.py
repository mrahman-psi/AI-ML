import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from _model import Net  # Make sure _model.py is in the same directory
# from velocityGAN import Generator  # Import the Generator class from velocityGAN.py
import os
import torch._classes  # Prevent Streamlit torch watcher crash
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ========== VelocityGAN Generator Definition ==========
class MinMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class Generator(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            MinMaxPool2d()
        )
        self.middle = nn.Sequential(
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.decoder(self.middle(self.encoder(x)))
    
# ========== Streamlit App ========== 
# Title
st.set_page_config(layout="wide")
st.title("Seismic Velocity Map Prediction App")

# Sidebar
st.sidebar.header("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload Seismic Waveform (.npy)", type=["npy"])
velocity_file = st.sidebar.file_uploader("(Optional) Upload Ground Truth Velocity Map (.npy)", type=["npy"])
model_type = st.sidebar.selectbox("Select Model", ["VelocityGAN", "InversionNet", "ConvNeXt U-Net"])

# Load ConvNeXt U-Net model
@st.cache_resource
def load_convnext_unet_model(path="best_model_1234_submodel0.pt"):
    model = Net(backbone='convnext_small', pretrained=False)
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.eval().float()
    return model

# Load VelocityGAN model
@st.cache_resource
def load_velocitygan_model(path="generator_thebest.pt"):
    model = Generator(in_channels=5)
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.eval().float()
    return model

# Dummy InversionNet loader (replace with real)
def load_inversionnet_model():
    st.warning("InversionNet model loading not implemented.")
    return lambda x: x

# Inference for ConvNeXt
def predict_with_convnext(model, input_waveform):
    with torch.no_grad():
        x = torch.tensor(input_waveform).float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif x.dim() == 3:
            x = x.unsqueeze(0)  # [1, C, H, W]
        output = model(x).squeeze().numpy()
    return output

# Inference for VelocityGAN
def predict_with_velocitygan(model, input_waveform):
    with torch.no_grad():
        x = torch.tensor(input_waveform).float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        output = model(x).squeeze().numpy()
    return output

# Main Prediction
if st.button("Predict") and uploaded_file is not None:
    waveform = np.load(uploaded_file)

    if model_type == "ConvNeXt U-Net":
        model = load_convnext_unet_model()
        prediction = predict_with_convnext(model, waveform)
    elif model_type == "VelocityGAN":
        model = load_velocitygan_model()
        prediction = predict_with_velocitygan(model, waveform)
    else:
        model = load_inversionnet_model()
        prediction = model(waveform)

    # Display
    fig, axs = plt.subplots(1, 3 if velocity_file else 1, figsize=(15, 5))
    axs = axs if isinstance(axs, np.ndarray) else [axs]

    # Handle prediction shape properly (no zoom)
    # Handle prediction shape robustly
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().detach().numpy()

    # Flatten prediction to 2D
    if prediction.ndim == 4:
        # Shape: (B, C, H, W)
        pred_img = prediction[0, 0]  # first image in batch
    elif prediction.ndim == 3:
        # Shape: (B, H, W) or (C, H, W)
        pred_img = prediction[0] if prediction.shape[0] == 1 else prediction[0]
    elif prediction.ndim == 2:
        # Shape: (H, W) → OK
        pred_img = prediction
    else:
        raise ValueError(f"Unsupported prediction shape for imshow: {prediction.shape}")

    axs[0].imshow(pred_img, cmap="jet", aspect="auto")
    axs[0].set_title("Predicted Velocity Map")

    if velocity_file:
        ground_truth = np.load(velocity_file)
        axs[1].imshow(ground_truth, cmap="jet", aspect="auto")
        axs[1].set_title("Ground Truth Velocity Map")

        diff = np.abs(pred_img - ground_truth)
        axs[2].imshow(diff, cmap="hot", aspect="auto")
        axs[2].set_title("Prediction Error Map")

    st.pyplot(fig)
    st.success("Prediction complete and displayed successfully!")
