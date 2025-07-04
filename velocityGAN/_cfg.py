# Configuration file
from types import SimpleNamespace
import torch

cfg= SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.local_rank = 0
cfg.seed = 123
cfg.subsample = None

cfg.backbone = "convnext_small.fb_in22k_ft_in1k"
cfg.ema = True
cfg.ema_decay = 0.99

cfg.epochs = 30
cfg.batch_size = 16
cfg.batch_size_val = 16


cfg.early_stopping = {"patience": 3, "streak": 0}
cfg.logging_steps = 100


cfg.input_size = (256, 256)
cfg.num_workers = 0
cfg.pretrained_weights = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/pretrainedModel/convnext_small.fb_in22k_ft_in1k.pth"
cfg.best_model_path = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/output/best_model_123.pth"
cfg.output_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/ConvNeXt/output/"
cfg.mmap_mode = "r"
cfg.world_size = 1  # For distributed training, set this to the number of GPUs
cfg.ensemble = True  # Whether to use ensemble predictions
cfg.ensemble_weights = [1.0]  # Weights for ensemble models, if applicable
cfg.ensemble_models = 5  # Number of models in the ensemble, if applicable

cfg.ensemble_weights = [w for w in cfg.ensemble_weights if w is not None]
cfg.TF_ENABLE_ONEDNN_OPTS=0



'''
_cfg.py – defines configurations like device, paths, hyperparams.
_dataset.py – handles dataset loading and preprocessing.
_model.py – defines the ConvNeXt model (Net) and ensemble logic.
_train.py – trains the model and saves .pth weights to pretrainedModel/.
preTrainedModel.py – loads these .pth weights into an ensemble for:
    validation.py – evaluates on a validation set.
    testPredict.py – performs prediction on test data.
'''