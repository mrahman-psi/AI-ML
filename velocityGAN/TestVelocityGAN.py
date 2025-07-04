import csv, glob, time, datetime
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import models                                   # your Generator is here

# ---------------------
# Device Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- utils --------------------
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

# -------------------- dataset ------------------
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

# -------------------- inference ----------------
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model_path = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/velocityGAN/outputs/generator_1.pth"
    model      = models.Generator()              # instantiate first
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # test files
    test_files = sorted(glob.glob("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test/*.npy"))
    test_dl = torch.utils.data.DataLoader(
        TestDataset(test_files),
        batch_size = 16,
        shuffle    = False,
        num_workers= 0
    )

    x_cols     = [f"x_{i}" for i in range(1,70,2)]
    fieldnames = ["oid_ypos"] + x_cols
    row_count  = 0
    t0         = time.time()

    outputs = None
    stems = None

    with open("submission-GAN.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        amp_ctx = torch.autocast("cuda") if device.type=="cuda" else nullcontext()
        with torch.inference_mode(), amp_ctx:
            for inputs, stems in tqdm(test_dl, total=len(test_dl)):
                inputs = inputs.to(device)                   # [B,C,H,W]
                outputs = model(inputs).squeeze(1)           # [B,H,W]

                # write predictions
                for pred, stem in zip(outputs.cpu().numpy(), stems):
                    for y in range(pred.shape[0]):           # H axis
                        row = {"oid_ypos": f"{stem}_y_{y}"}
                        row.update({f"x_{i}": pred[y,i] for i in range(1,70,2)})
                        writer.writerow(row)
                        row_count += 1
                    if row_count % 100_000 == 0:
                        csvfile.flush()

    print(f"Total predictions written: {row_count}")
    print(f"Elapsed time: {format_time(time.time()-t0)}")
    return outputs, stems

# -------------------- visual -------------------
def plot_predictions(outputs, stems):
    fig, axes = plt.subplots(3,5, figsize=(10,6))
    axes = axes.flatten()
    for ax, img, stem in zip(axes, outputs[:len(axes)], stems):
        ax.imshow(img.cpu(), cmap="gray")
        ax.set_title(stem); ax.axis("off")
    for ax in axes[len(outputs):]: ax.axis("off")
    plt.tight_layout(); plt.show()

# -------------------- main ---------------------
if __name__ == "__main__":
    outs, stems = run_inference()
    plot_predictions(outs, stems)
