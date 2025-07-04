import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import Generator
from dataset import VelocityGANDataset
import matplotlib.pyplot as plt


# ---------------------
# Device Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def evaluate_generator(G, dataloader, num_batches=3):
    G.eval()
    mse_loss = torch.nn.MSELoss()
    mae_total, mse_total, count = 0, 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = G(x)
            mae_total += F.l1_loss(pred, y, reduction='mean').item()
            mse_total += mse_loss(pred, y).item()
            count += 1
            if i >= num_batches - 1:
                break

    print(f"\nEvaluation over {count} batches:")
    print(f"  MAE : {mae_total / count:.4f}")
    print(f"  MSE : {mse_total / count:.4f}")


def predict_on_test(generator_model_path, test_dir, output_dir="predictions", img_size=(128, 128), extension="csv"):
    os.makedirs(output_dir, exist_ok=True)

    G = Generator(in_channels=5).to(device)
    G.load_state_dict(torch.load(generator_model_path, map_location=device))
    G.eval()

    all_predictions = []
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.npy')]

    for fname in tqdm(test_files, desc="Predicting on test set"):
        seismic = np.load(os.path.join(test_dir, fname))
        if seismic.ndim == 4:
            x = torch.tensor(seismic, dtype=torch.float32).permute(3, 2, 0, 1)[:, 0, :, :]
        elif seismic.ndim == 3:
            x = torch.tensor(seismic, dtype=torch.float32).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported shape {seismic.shape} in file {fname}")

        x = F.interpolate(x.unsqueeze(0), size=img_size, mode="bilinear", align_corners=False).to(device)

        with torch.no_grad():
            # x: shape (1, 70, 128, 128)
            if x.shape[1] > 5:
                x = x[:, :5, :, :]  # take only first 5 channels
            elif x.shape[1] < 5:
                raise ValueError(f"Expected 5 channels, got {x.shape[1]}")
            pred = G(x).squeeze().cpu().numpy()

        row = [os.path.splitext(fname)[0]] + pred.flatten().tolist()
        all_predictions.append(row)

    df = pd.DataFrame(all_predictions)
    df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False, header=False)

    print(f"All predictions saved to: {os.path.join(output_dir, 'all_predictions.csv')}")


def render_velocity_maps(csv_path, output_dir="rendered_maps", img_size=(128, 128), cmap="viridis"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, header=None)

    for i, row in df.iterrows():
        sample_id = row[0]
        flat_data = row[1:].values.astype(np.float32)

        if flat_data.size != img_size[0] * img_size[1]:
            print(f"Skipping {sample_id}: incorrect data size {flat_data.size}")
            continue

        velocity_map = flat_data.reshape(img_size)

        plt.figure(figsize=(4, 4))
        plt.imshow(velocity_map, cmap=cmap, aspect='auto')
        plt.title(f"Velocity Map: {sample_id}")
        plt.axis('off')

        output_path = os.path.join(output_dir, f"{sample_id}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    print(f"Rendering completed. Images saved to '{output_dir}'.")


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="Path to trained generator .pth file")
    # parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test .npy files")
    # parser.add_argument("--output_dir", type=str, default="predictions", help="Output directory for CSV/images")
    # parser.add_argument("--render", action="store_true", help="Render velocity maps as PNGs")
    # args = parser.parse_args()
    start_time = time.time()
    model = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/outputs/generator_thebest.pth"
    test_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test/"
    output_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/predictions/"
    render = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/runs/store_true/"

    predict_on_test(model, test_dir, output_dir=output_dir)

    if render:
        csv_path = os.path.join(output_dir, "all_predictions.csv")
        render_velocity_maps(csv_path, output_dir=os.path.join(output_dir, "images"))
    
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
#python inference.py --model outputs/generator.pth --test_dir /path/to/test --output_dir test_results --render
 
