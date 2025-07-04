import os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
import _cfg as cfg
import dataset as ds

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- MinMax Pool --------------------
class MinMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x):
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        min_pool = -F.max_pool2d(-x, self.kernel_size, self.stride, self.padding)
        return 0.5 * (max_pool + min_pool)

# -------------------- Residual Block --------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

# -------------------- Generator --------------------
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

# -------------------- Discriminator --------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------- InversionNet --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class InversionNet(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.pool = MinMaxPool2d()
        self.drop = nn.Dropout2d(0.3)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.drop(self.bottleneck(self.pool(e3)))

        up3 = self.up3(b)
        if up3.shape[-2:] != e3.shape[-2:]:
            up3 = F.interpolate(up3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.drop(self.dec3(torch.cat([up3, e3], dim=1)))

        up2 = self.up2(d3)
        if up2.shape[-2:] != e2.shape[-2:]:
            up2 = F.interpolate(up2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.drop(self.dec2(torch.cat([up2, e2], dim=1)))

        up1 = self.up1(d2)
        if up1.shape[-2:] != e1.shape[-2:]:
            up1 = F.interpolate(up1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        out = self.out(d1)

        # ✅ Resize to match label (usually [70, 70])
        out = F.interpolate(out, size=(70, 70), mode='bilinear', align_corners=False)

        return out


# -------------------- Evaluation --------------------
def evaluate(model, loader, name="Validation"):
    model.eval()
    total_mae, total_mse, count = 0, 0, 0
    with torch.no_grad():
        for seis, vel in loader:
            seis, vel = seis.to(device), vel.to(device)
            pred = model(seis)
            total_mae += F.l1_loss(pred, vel).item()
            total_mse += F.mse_loss(pred, vel).item()
            count += 1
    print(f"\n{name} MAE: {total_mae / count:.2f} | MSE: {total_mse / count:.2f}")
    return total_mae / count

# -------------------- Training Entry Point --------------------
def train():
    #data_dir = cfg.cfg.input_path  # Or manually set the path here
    data_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/train_samples/"
    #test_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test/"
    output_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/outputs"
    log_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/runs/velocitygan"

    #output_dir = cfg.cfg.output_dir
    #log_dir = os.path.join(output_dir, "logs")
    batch_size, lr, epochs = 16, 0.0002, 10

    writer = SummaryWriter(log_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_ds = ds.VelocityGANDataset(data_dir)
    valid_ds = ds.VelocityGANDataset(data_dir)


    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=0)

    G, D, I = Generator().to(device), Discriminator().to(device), InversionNet().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_I = torch.optim.Adam(I.parameters(), lr=5e-4)

    scheduler_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=5, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=5, gamma=0.5)
    scheduler_I = torch.optim.lr_scheduler.StepLR(opt_I, step_size=5, gamma=0.5)

    best_mae = float("inf")
    for epoch in range(epochs):
        G.train(); D.train(); I.train()
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (seis, vel) in enumerate(loop):
            seis, vel = seis.to(device), vel.to(device)
            bs = seis.size(0)
            real, fake = torch.ones(bs, 1).to(device), torch.zeros(bs, 1).to(device)

            # Discriminator
            opt_D.zero_grad()
            loss_D = F.mse_loss(D(vel).view(bs, -1).mean(1, keepdim=True), real) + \
                     F.mse_loss(D(G(seis).detach()).view(bs, -1).mean(1, keepdim=True), fake)
            loss_D.backward(); opt_D.step()

            # Generator
            opt_G.zero_grad()
            loss_G = F.mse_loss(D(G(seis)).view(bs, -1).mean(1, keepdim=True), real)
            loss_G.backward(); opt_G.step()

            # InversionNet
            opt_I.zero_grad()
            loss_I = F.mse_loss(I(seis), vel)
            loss_I.backward(); opt_I.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item(), loss_I=loss_I.item())

        # Validation
        val_mae = evaluate(G, valid_dl)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(G.state_dict(), os.path.join(output_dir, "generator_best.pth"))
            print(f"\U0001F4CC New best model saved with MAE: {best_mae:.2f}")

        evaluate(I, valid_dl, name="InversionNet")

        scheduler_G.step()
        scheduler_D.step()
        scheduler_I.step()

    print("\n✅ Training complete.")
    writer.close()
    print(f"Models saved to {output_dir}")


if __name__ == "__main__":
    train()
