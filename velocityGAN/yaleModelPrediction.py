import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------------------
# Device Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print start time
start_time = time.time()
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

# ---------------------
# Generator Model
# ---------------------
class Generator(nn.Module):
    def __init__(self, in_channels=5):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = self.final(x)
        # Optional: scale output to [1500, 4500]
        x = (x + 1) * 1500 + 1500
        return x

# ---------------------
# InversionNet Model
# ---------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ---------------------
# InversionNet Model
# ---------------------

class InversionNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(InversionNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.up3(bottleneck)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))
        return self.out_conv(dec1)

# ---------------------
# Discriminator Model
# ---------------------
class Discriminator(nn.Module):
    def __init__(self,  input_shape=(1, 128, 128), in_channels=1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class MinMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MinMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        min_pool = -F.max_pool2d(-x, self.kernel_size, self.stride, self.padding)
        return 0.5 * (max_pool + min_pool)  # Combine mean-wise


# ---------------------
# Training Loop for InversionNet
# ---------------------
def train_inversionnet(G, dataloader, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    G.train()
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"InversionNet Epoch [{epoch+1}/{num_epochs}]")
        for seismic_data, velocity_data in loop:
            seismic_data, velocity_data = seismic_data.to(device), velocity_data.to(device)
            optimizer.zero_grad()
            outputs = G(seismic_data)
            loss = criterion(outputs, velocity_data)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

# ---------------------
# Custom Dataset
# ---------------------
class VelocityGANDataset(Dataset):
    def __init__(self, traindf, train_dir, ftscale=True):
        self.traindf = traindf
        self.train_dir = train_dir
        self.ftscale = ftscale
        self.file_cache = {}

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, index):
        row = self.traindf.iloc[index]
        veltype, ifile, isample = row["veltype"], int(row["ifile"]), int(row["isample"])

        if ("Vel" in veltype) or ("Style" in veltype):
            data_file = os.path.join(self.train_dir, veltype, "data", f"data{ifile}.npy")
            model_file = os.path.join(self.train_dir, veltype, "model", f"model{ifile}.npy")
        else:
            fault_num = 2 * ifile + 4 * ("_B" in veltype)
            data_file = os.path.join(self.train_dir, veltype, f"seis{fault_num}_1_0.npy")
            model_file = os.path.join(self.train_dir, veltype, f"vel{fault_num}_1_0.npy")

        if data_file not in self.file_cache:
            data = np.load(data_file)
            if self.ftscale:
                for itime in range(data.shape[2]):
                    data[:, :, itime, :] *= (1.0 + (itime / 200) ** 1.5)
            self.file_cache[data_file] = data

        if model_file not in self.file_cache:
            self.file_cache[model_file] = np.load(model_file)

        seismic = self.file_cache[data_file][isample]
        velocity = self.file_cache[model_file][isample]

        seismic_tensor = torch.tensor(seismic, dtype=torch.float32)
        velocity_tensor = torch.tensor(velocity, dtype=torch.float32)

        if seismic_tensor.ndim == 4:
            h, w, t, c = seismic_tensor.shape # (num_sources, num_receivers, time_steps, channels)
            seismic_tensor = seismic_tensor.permute(3, 2, 0, 1)[:, 0, :, :] # (channels, time_steps, num_sources, num_receivers)

        if seismic_tensor.ndim == 2:
            seismic_tensor = seismic_tensor.unsqueeze(0)
        if velocity_tensor.ndim == 2:
            velocity_tensor = velocity_tensor.unsqueeze(0)

        seismic_tensor = F.interpolate(seismic_tensor.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False).squeeze(0)
        velocity_tensor = F.interpolate(velocity_tensor.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False).squeeze(0)

        return seismic_tensor, velocity_tensor

# ---------------------
# DataLoader
# ---------------------
def get_dataloader(traindf, train_dir, batch_size=16, shuffle=True):
    dataset = VelocityGANDataset(traindf, train_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available(), num_workers=0)

# ---------------------
# Training Loop
# ---------------------
def train(G, D, dataloader, num_epochs=10, lr=0.0002):
    # Replace BCE loss with MSE (LSGAN)
    criterion = nn.MSELoss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)

    
    G.train()
    D.train()

    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for seismic_data, velocity_data in loop:
            seismic_data, velocity_data = seismic_data.to(device), velocity_data.to(device)
            batch_size = seismic_data.size(0)

            optimizer_D.zero_grad()
            # For real labels
            real_labels = torch.ones((batch_size, 1), device=device)
            # For fake labels
            fake_labels = torch.zeros((batch_size, 1), device=device)

            outputs_real = D(velocity_data)
            outputs_fake = D(G(seismic_data).detach())
            # Flatten or average discriminator outputs to match label shape
            outputs_real = outputs_real.view(batch_size, -1).mean(dim=1, keepdim=True)
            outputs_fake = outputs_fake.view(batch_size, -1).mean(dim=1, keepdim=True)
            loss_D = criterion(outputs_real, real_labels) + criterion(outputs_fake, fake_labels)
            optimizer_G.zero_grad()
            outputs = D(G(seismic_data))
            outputs = outputs.view(batch_size, -1).mean(dim=1, keepdim=True)
            loss_G = criterion(outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()
            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

# ---------------------
# Evaluation
# ---------------------
def evaluate_generator(G, dataloader, num_batches=3):
    G.eval()
    mse_loss = nn.MSELoss()
    mae_total, mse_total, count = 0, 0, 0

    with torch.no_grad():
        for i, (seis, vel) in enumerate(dataloader):
            seis, vel = seis.to(device), vel.to(device)
            pred = G(seis)
            mae_total += F.l1_loss(pred, vel, reduction='mean').item()
            mse_total += mse_loss(pred, vel).item()
            count += 1
            if i >= num_batches - 1:
                break

    print(f"\nEvaluation over {count} batches:")
    print(f"  MAE : {mae_total / count:.4f}")
    print(f"  MSE : {mse_total / count:.4f}")

# ---------------------
# Predict on Test Set
# ---------------------
def predict_on_test(generator_model_path, test_dir, output_dir="predictions", extension="csv", model_type=None):
    os.makedirs(output_dir, exist_ok=True)

    # Determine model type
    if model_type is None:
        if "inversionnet" in generator_model_path.lower():
            model_type = "inversionnet"
        elif "generator" in generator_model_path.lower():
            model_type = "generator"
        else:
            raise ValueError("Cannot determine model_type from model path. Please specify model_type explicitly.")

    # Load model
    if model_type == "generator":
        G = Generator(in_channels=5).to(device)
    elif model_type == "inversionnet":
        G = InversionNet(in_channels=5).to(device)
    else:
        raise ValueError("Unknown model_type. Use 'generator' or 'inversionnet'.")

    G.load_state_dict(torch.load(generator_model_path))
    G.eval()

    all_predictions = []
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.npy')]

    for fname in tqdm(test_files, desc="Predicting on test set"):
        seismic = np.load(os.path.join(test_dir, fname))
        if seismic.ndim == 4:
            h, w, t, c = seismic.shape
            seismic_tensor = torch.tensor(seismic, dtype=torch.float32).permute(3, 2, 0, 1)[:, 0, :, :]
        elif seismic.ndim == 3:
            seismic_tensor = torch.tensor(seismic, dtype=torch.float32).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported shape {seismic.shape} in file {fname}")

        in_channels = 5
        if seismic_tensor.shape[0] > in_channels:
            seismic_tensor = seismic_tensor[:in_channels, :, :]
        elif seismic_tensor.shape[0] < in_channels:
            raise ValueError(f"Expected at least {in_channels} channels, got {seismic_tensor.shape[0]}")

        seismic_tensor = F.interpolate(seismic_tensor.unsqueeze(0), size=(128, 128),
                                       mode="bilinear", align_corners=False).squeeze(0).to(device)

        with torch.no_grad():
            pred_velocity = G(seismic_tensor.unsqueeze(0)).squeeze().cpu().numpy()

        # Flatten and append with filename
        row = [os.path.splitext(fname)[0]] + pred_velocity.flatten().tolist()
        all_predictions.append(row)

    # Build DataFrame and write once
    all_df = pd.DataFrame(all_predictions)
    all_df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False, header=False)

    print(f"All predictions saved to: {os.path.join(output_dir, 'all_predictions.csv')}")

# Following code is commented out to avoid execution in the main script. This function creates a one to one mapping of predictions to CSV files.
'''
def predict_on_test(generator_model_path, test_dir, output_dir="predictions", extension="csv", model_type=None):
    os.makedirs(output_dir, exist_ok=True)

    # Dynamically determine model_type if not provided
    if model_type is None:
        if "inversionnet" in generator_model_path.lower():
            model_type = "inversionnet"
        elif "generator" in generator_model_path.lower():
            model_type = "generator"
        else:
            raise ValueError("Cannot determine model_type from model path. Please specify model_type explicitly.")

    if model_type == "generator":
        G = Generator(in_channels=5).to(device)
    elif model_type == "inversionnet":
        G = InversionNet(in_channels=5).to(device)
    else:
        raise ValueError("Unknown model_type. Use 'generator' or 'inversionnet'.")

    G.load_state_dict(torch.load(generator_model_path))
    G.eval()

    test_files = [f for f in os.listdir(test_dir) if f.endswith('.npy')]
    for fname in tqdm(test_files, desc="Predicting on test set"):
        seismic = np.load(os.path.join(test_dir, fname))
        if seismic.ndim == 4:
            h, w, t, c = seismic.shape
            seismic_tensor = torch.tensor(seismic, dtype=torch.float32).permute(3, 2, 0, 1)[:, 0, :, :]
        elif seismic.ndim == 3:
            seismic_tensor = torch.tensor(seismic, dtype=torch.float32).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported shape {seismic.shape} in file {fname}")

        in_channels = 5
        if seismic_tensor.shape[0] > in_channels:
            seismic_tensor = seismic_tensor[:in_channels, :, :]
        elif seismic_tensor.shape[0] < in_channels:
            raise ValueError(f"Expected at least {in_channels} channels, got {seismic_tensor.shape[0]}")

        seismic_tensor = F.interpolate(seismic_tensor.unsqueeze(0), size=(128, 128), 
                                       mode="bilinear", align_corners=False).squeeze(0).to(device)

        with torch.no_grad():
            pred_velocity = G(seismic_tensor.unsqueeze(0)).squeeze().cpu().numpy()

        base_name = os.path.splitext(fname)[0]
        if extension == "csv":
            pd.DataFrame(pred_velocity).to_csv(os.path.join(output_dir, f"{base_name}_pred.csv"), index=False, header=False)
        else:
            np.save(os.path.join(output_dir, f"{base_name}_pred.npy"), pred_velocity)

    print(f"Predictions saved in '{output_dir}'")
'''
# Rendering Velocity map as CSV

def render_velocity_maps_from_csv(csv_path, output_dir="rendered_maps", img_size=(128, 128), cmap="viridis", save_images=True):
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

        if save_images:
            output_path = os.path.join(output_dir, f"{sample_id}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        else:
            plt.show()
        break # Remove this line to render all images, currently only renders the first one    
    # End of loop
    
    print(f"Rendering completed. Images saved to '{output_dir}'.")



# ---------------------
# Metadata Generator
# ---------------------
def get_traindf():
    veltypes = ["FlatVel", "FlatFault", "CurveVel", "CurveFault", "Style"]
    veltype, ifile, isample = [], [], []
    for vt in veltypes:
        for ab in ["_A", "_B"]:
            for f in [1, 2]:
                for s in range(500):
                    veltype.append(vt + ab)
                    ifile.append(f)
                    isample.append(s)
    return pd.DataFrame({"veltype": veltype, "ifile": ifile, "isample": isample})

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    # Load metadata and split
    print("Generating training DataFrame...")
    metadata = get_traindf()
    metadata.to_csv("train_metadata.csv", index=False)
    print("Loading training metadata...")
    traindf = pd.read_csv("train_metadata.csv")
    train_df, val_df = train_test_split(traindf, test_size=0.1, random_state=42)
    train_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/train_samples/"
    test_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test/"

    # Choose model: "inversionnet" or "generator"
    #model_type = "inversionnet"  # or "generator"
    model_type = "generator"  # or "inversionnet"

    # Setup loaders
    train_loader = get_dataloader(train_df, train_dir, batch_size=8)
    val_loader = get_dataloader(val_df, train_dir, batch_size=8)

    # Infer input shape
    sample_seis, _ = next(iter(train_loader))
    in_channels = sample_seis.shape[1]
    input_shape = (1, 128, 128)

    # Sanity check
    if model_type not in ["inversionnet", "generator"]:
        raise ValueError("model_type must be 'inversionnet' or 'generator'")
    
    print(f"Using model type: {model_type}")

    # -------------------------
    # Train and Save Model
    # -------------------------
    if model_type == "inversionnet":
        G = InversionNet(in_channels=in_channels).to(device)
        print("Training InversionNet...")
        train_inversionnet(G, train_loader, num_epochs=10, lr=0.001)
        torch.save(G.state_dict(), "inversionnet.pth")
    else:  # generator
        G = Generator(in_channels=in_channels).to(device)
        D = Discriminator(input_shape=input_shape).to(device)
        print("Training VelocityGAN...")
        train(G, D, train_loader, num_epochs=10, lr=0.0002)
        torch.save(G.state_dict(), "generator.pth")
        torch.save(D.state_dict(), "discriminator.pth")

    # -------------------------
    # Evaluate
    # -------------------------
    print("Evaluating model...")
    evaluate_generator(G, val_loader)

    # -------------------------

    # Predict on Test
    # -------------------------
    print("Generating test predictions...")
    weight_path = "inversionnet.pth" if model_type == "inversionnet" else "generator.pth"
    predict_on_test(weight_path, test_dir, output_dir="test_predictions_csv", extension="csv", model_type=model_type)

    # -------------------------
    # Render Velocity Maps
    # -------------------------
    # print("Rendering velocity maps from predictions...")
    # render_velocity_maps_from_csv("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test_predictions_csv_All_GAN/all_predictions.csv", 
    #                               output_dir="rendered_maps" )
    # #render_velocity_maps_from_csv("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test_predictions_csv_GAN/0000fd8ec8_pred.csv", output_dir="rendered_maps")

    # print("Velocity maps rendered successfully.")



    # Done
    end_time = time.time()
    print(f"Training and evaluation completed in {end_time - start_time:.2f} seconds.")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
