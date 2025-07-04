import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from _cfg import cfg
from _model import Net, EnsembleModel

# Define RUN_VALID and RUN_TEST flags
RUN_VALID = False
RUN_TEST = False

if RUN_VALID or RUN_TEST:

    # Load pretrained models
    models = []
    for f in sorted(glob.glob("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/pretrainedModel/*.pth")):
        print("Loading: ", f)
        m = Net(
            backbone="convnext_small.fb_in22k_ft_in1k",
            pretrained=False,
        )
        state_dict= torch.load(f, map_location=cfg.device, weights_only=True)
        state_dict= {k.removeprefix("_orig_mod."):v for k,v in state_dict.items()} # Remove torch.compile() prefix

        m.load_state_dict(state_dict)
        models.append(m)
    
    # Combine
    model = EnsembleModel(models)
    model = model.to(cfg.device)
    model = model.eval()
    print("n_models: {:_}".format(len(models)))