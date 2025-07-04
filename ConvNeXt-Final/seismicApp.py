import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from _model import Net  # Make sure _model.py is in the same directory
import os
import torch._classes  # Prevent Streamlit torch watcher crash
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

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

# Dummy GAN/InversionNet loaders (replace with real)
def load_velocitygan_model():
    st.warning("VelocityGAN model loading not implemented.")
    return lambda x: x

def load_inversionnet_model():
    st.warning("InversionNet model loading not implemented.")
    return lambda x: x

# Inference with ConvNeXt
def predict_with_convnext(model, input_waveform):
    with torch.no_grad():
        x = torch.tensor(input_waveform).float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif x.dim() == 3:
            x = x.unsqueeze(0)  # [1, C, H, W]
        output = model(x).squeeze().numpy()  # [H, W] or [5, H, W]
    return output

# Main Prediction
if st.button("Predict") and uploaded_file is not None:
    waveform = np.load(uploaded_file)

    if model_type == "ConvNeXt U-Net":
        model = load_convnext_unet_model()
        prediction = predict_with_convnext(model, waveform)
    elif model_type == "VelocityGAN":
        model = load_velocitygan_model()
        prediction = model(waveform)
    else:
        model = load_inversionnet_model()
        prediction = model(waveform)

    # Normalize prediction shape
    if prediction.ndim == 3:
        pred_img = prediction.mean(axis=0)  # (H, W)
    elif prediction.ndim == 2:
        pred_img = prediction
    else:
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")

    # Upsample to higher resolution for display
    pred_tensor = torch.tensor(pred_img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    pred_hr = F.interpolate(pred_tensor, scale_factor=2, mode='bilinear', align_corners=False)
    pred_img_highres = pred_hr.squeeze().numpy()  # [H*2, W*2]

    # Plot
    fig, axs = plt.subplots(1, 3 if velocity_file else 1, figsize=(18, 6))
    axs = axs if isinstance(axs, np.ndarray) else [axs]

    axs[0].imshow(pred_img_highres, cmap="jet", aspect="auto")
    axs[0].set_title("Predicted Velocity Map (Upscaled 2x)")

    if velocity_file:
        ground_truth = np.load(velocity_file)
        axs[1].imshow(ground_truth, cmap="jet", aspect="auto")
        axs[1].set_title("Ground Truth Velocity Map")

        # Resize ground truth if needed for shape match
        if ground_truth.shape != pred_img.shape:
            gt_tensor = torch.tensor(ground_truth).unsqueeze(0).unsqueeze(0).float()
            gt_resized = F.interpolate(gt_tensor, size=pred_img.shape, mode='bilinear', align_corners=False)
            ground_truth = gt_resized.squeeze().numpy()

        diff = np.abs(pred_img - ground_truth)
        axs[2].imshow(diff, cmap="hot", aspect="auto")
        axs[2].set_title("Prediction Error Map")

    print("Prediction shape:", prediction.shape)
    st.pyplot(fig)
    st.success("Prediction complete and displayed successfully!")



# execute the app by running the following command in terminal from the directory where this script is located:
#cd ConvNeXt-Final
# streamlit run seismicApp.py