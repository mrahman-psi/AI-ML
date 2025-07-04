import os
import numpy as np
import torch
import pandas as pd
import glob
from torch.utils.data import Dataset
from tqdm import tqdm

class VelocityGANDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        print(f"Loading {self.mode} data from {self.data_dir}")
        self.data, self.labels, self.records = self.load_metadata()

    def load_metadata(self):
        print("Loading metadata...")
        df= pd.read_csv("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/Input/folds.csv")
        
        print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        print(f"First 10 rows:\n{df.head(10)}")

        if self.mode == "train":
            df = df[df["fold"] != 0]
        else:
            df = df[df["fold"] == 0]

        data, labels, records = [], [], []
        mmap_mode = "r"

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {self.mode} data"):
            row = row.to_dict()
            # Hacky way to get exact file name
            p1 = os.path.join("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/Input/openfwi_float16_1/", row["data_fpath"])
            p2 = os.path.join("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/Input/openfwi_float16_1/", 
                              row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            p3 = os.path.join("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/Input/openfwi_float16_2/", row["data_fpath"])
            p4 = os.path.join("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/Input/openfwi_float16_2/", 
                              row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            farr= glob.glob(p1) + glob.glob(p2) + glob.glob(p3) + glob.glob(p4)
        
            # Map to lbl fpath
            farr= farr[0]
            flbl= farr.replace('seis', 'vel').replace('data', 'model')

            if not os.path.exists(flbl):
                raise FileNotFoundError(f"No velocity label found for {flbl}")

            arr = np.load(farr, mmap_mode=mmap_mode)
            lbl = np.load(flbl, mmap_mode=mmap_mode)

            data.append(arr)
            labels.append(lbl)
            records.append(row["dataset"])

        return data, labels, records

    def __getitem__(self, idx):
        row_idx = idx // 500
        col_idx = idx % 500

        x = self.data[row_idx][col_idx, ...]
        y = self.labels[row_idx][col_idx, ...]

        if self.mode == "train":
            if np.random.rand() < 0.5:
                x = x[::-1, :, ::-1]
                y = y[..., ::-1]

        # Ensure tensors are float32 to prevent Half/Float mismatch errors
        x = torch.tensor(x.copy(), dtype=torch.float32)
        y = torch.tensor(y.copy(), dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.records) * 500
