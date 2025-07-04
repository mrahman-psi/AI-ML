import csv
import time
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from _cfg import cfg
from _model import Net
from _utils import format_time

# -----------------------------
# Dataset Definition
# -----------------------------
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_files):
        self.test_files = test_files
    def __len__(self): return len(self.test_files)
    def __getitem__(self, i):
        f = self.test_files[i]
        stem = f.rsplit("\\",1)[-1].split(".")[0]
        x = np.load(f).astype(np.float32)        # (C,H,W)
        # Example stem: "test\000039dca2_y_3"
        # Split into prefix, id, and y value
        if "_y_" in stem:
            prefix_id, y_val = stem.rsplit("_y_", 1)
            prefix, id_val = prefix_id.split("\\", 1) if "\\" in prefix_id else ("", prefix_id)
            # prefix: "test", id_val: "000039dca2", y_val: "3"
            return torch.tensor(x), (prefix, id_val, int(y_val))
        else:
            return torch.tensor(x), stem
# -----------------------------
# Inference and Submission
# -----------------------------
def run_inference():
    model = Net(backbone='convnext_small', pretrained=False)
    model.load_state_dict(torch.load("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt-Final/output/best_model_1234_submodel0.pt", map_location='cpu'))
    # model = Net("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt-Final/output/best_model_1234_submodel1.pt")
    # model.load_state_dict(torch.load(cfg.best_model_path, map_location=cfg.device))
    model = model.to(cfg.device).float()  # Ensure model is in float32
    model.eval()

    ss = pd.read_csv("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/sample_submission.csv")
    row_count = 0
    t0 = time.time()

    test_files = sorted(glob.glob("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test/*.npy"))
    x_cols = [f"x_{i}" for i in range(1, 70, 2)]
    fieldnames = ["oid_ypos"] + x_cols

    test_ds = TestDataset(test_files)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=0  # safer for Windows
    )

    last_outputs, last_oids_test = None, None

    with open("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project//ConvNeXt/output/submission15.csv", "wt", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        inference_ctx = torch.autocast(cfg.device.type) if cfg.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), inference_ctx:
            for inputs, oids_test in tqdm(test_dl, total=len(test_dl)):
                inputs = inputs.to(cfg.device).float()  # Ensure inputs are float32
                # Ensure inputs is 4D for Conv2D and pad
                if inputs.dim() == 5 and inputs.size(1) == 1:
                    inputs = inputs.squeeze(1)  # [B, 1, C, H, W] → [B, C, H, W]

                outputs = model(inputs.float())  # [B, 1, H, W]
                outputs = outputs.squeeze(1)  # [B, H, W]

                last_outputs = outputs
                last_oids_test = oids_test

                for y_pred, oid_test in zip(outputs.cpu().numpy(), oids_test):
                    for y_pos in range(y_pred.shape[0]):
                        row = {
                            "oid_ypos": f"{str(oid_test)}_y_{y_pos}"
                        }
                        row.update({
                            f"x_{i}": y_pred[y_pos, i] for i in range(1, 70, 2)
                        })
                        writer.writerow(row)
                        row_count += 1

                        if row_count % 100_000 == 0:
                            csvfile.flush()

    print(f"Total predictions written: {row_count}")
    print(f"Elapsed time: {format_time(time.time() - t0)}")
    return last_outputs, last_oids_test

# -----------------------------
# Visualization
# -----------------------------
def plot_predictions(outputs, oids_test):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    axes = axes.flatten()

    n = min(len(outputs), len(axes))
    for i in range(n):
        img = outputs[i].cpu().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(oids_test[i])
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    from contextlib import nullcontext  # for CPU autocast fallback

    outputs, oids_test = run_inference()

    if outputs is not None and oids_test is not None:
        plot_predictions(outputs, oids_test)
