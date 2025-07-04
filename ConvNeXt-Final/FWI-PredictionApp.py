import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from _model import Net

def load_model():
    model = Net(backbone="convnext_small", pretrained=False)
    state_dict = torch.load("output/best_model_1234_submodel0.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_with_convnext(model, waveform):
    # waveform shape: [H, W] => [1, C, H, W]
    x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
    if x.shape[2] != 72:
        x = torch.nn.functional.interpolate(x, size=(72, 72), mode="bilinear", align_corners=False)
    x = x.expand(-1, 5, -1, -1)  # convert to [1, 5, H, W] as expected by the model
    with torch.no_grad():
        output = model(x)
    return output.squeeze().numpy()

def display_velocity_map(map_data, title="Velocity Map"):
    plt.figure(figsize=(10, 4))
    plt.imshow(map_data, cmap="jet", aspect="auto")
    plt.colorbar(label="Velocity (m/s)")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    st.pyplot(plt.gcf())
    plt.close()

def main():
    st.title("Seismic Velocity Map Prediction")
    st.markdown("Upload a waveform `.npy` file to generate its velocity map.")

    waveform_file = st.file_uploader("Upload waveform file (.npy)", type=["npy"])
    velocity_file = st.file_uploader("(Optional) Upload ground truth velocity map (.npy)", type=["npy"])

    if waveform_file is not None:
        waveform = np.load(waveform_file)
        st.write(f"Waveform shape: {waveform.shape}")
        st.subheader("Uploaded Waveform")
        plt.figure(figsize=(10, 4))
        plt.imshow(waveform, cmap="gray", aspect="auto")
        plt.colorbar()
        st.pyplot(plt.gcf())
        plt.close()

        model = load_model()
        prediction = predict_with_convnext(model, waveform)
        display_velocity_map(prediction, title="Predicted Velocity Map")

        if velocity_file is not None:
            velocity_gt = np.load(velocity_file)
            st.subheader("Ground Truth Velocity Map")
            display_velocity_map(velocity_gt, title="Ground Truth Velocity Map")

            mae = np.mean(np.abs(prediction - velocity_gt))
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

if __name__ == "__main__":
    main()

## streamlit run FWI-PredictionApp.py