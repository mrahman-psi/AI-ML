# Load the new model and do the model validation
import torch
import torch.nn as nn
from _cfg import cfg
from _model import Net, ModelEMA, EnsembleModel
from _utils import format_time
from torch.utils.data import DataLoader
from torch import amp
from _dataset import CustomDataset
from tqdm import tqdm
#from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast


# Load ensemble state_dict
state_dict = torch.load("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt-Final/output/best_model_123.pt", map_location="cpu")

# Extract keys from 'models.0.' only
submodel_0_state_dict = {
    k.replace("models.0.", ""): v
    for k, v in state_dict.items()
    if k.startswith("models.0.")
}
# Save as a standalone checkpoint
torch.save(submodel_0_state_dict, "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt-Final/output/best_model_1234_submodel0.pt")

submodel_1_state_dict = {
    k.replace("models.1.", ""): v
    for k, v in state_dict.items()
    if k.startswith("models.1.")
}
torch.save(submodel_1_state_dict, "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt-Final/output/best_model_1234_submodel1.pt")

# Initialize the same architecture (use correct backbone!)
#model = Net(backbone='convnext_base', pretrained=False)
model = Net(backbone='convnext_small', pretrained=False)

# Load the submodel weights
model.load_state_dict(torch.load("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt-Final/output/best_model_1234_submodel0.pt", map_location="cpu"), strict=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

valid_ds = CustomDataset(cfg=cfg, mode="valid")
valid_dl = torch.utils.data.DataLoader(
    valid_ds,
    sampler=torch.utils.data.SequentialSampler(valid_ds),
    batch_size=cfg.batch_size_val,
    num_workers=0,  # safer for Windows
)

criterion = nn.L1Loss()
val_logits = []
val_targets = []

with torch.no_grad():
    for x, y in tqdm(valid_dl):
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        
        with amp.autocast_mode.autocast(device_type=cfg.device.type, enabled=True):
            out = model(x)

        val_logits.append(out.cpu())
        val_targets.append(y.cpu())

val_logits = torch.cat(val_logits, dim=0)
val_targets = torch.cat(val_targets, dim=0)
total_loss = criterion(val_logits, val_targets).item()

print("=" * 25)
print("Val MAE: {:.2f}".format(total_loss))
print("=" * 25)
