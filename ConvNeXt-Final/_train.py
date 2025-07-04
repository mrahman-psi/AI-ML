import os
import time 
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from _cfg import cfg
from _dataset import CustomDataset
from _model import ModelEMA, Net, EnsembleModel
from _utils import format_time
import glob

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup(rank=0, world_size=1):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        print("DDP initialized.")
        return True
    else:
        print("Single GPU detected. Skipping DDP.")
        return False

def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
def main(cfg, is_ddp):

    # ========== Datasets / Dataloaders ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Loading data..")
    train_ds = CustomDataset(cfg=cfg, mode="train")
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if is_ddp else None
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle=(train_sampler is None)
    )

    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank) if is_ddp else None
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        sampler=valid_sampler,
        batch_size=cfg.batch_size_val,
        num_workers=0,
        shuffle=False
    )

    # ========== Model / Optim ==========
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
    model= model.to(cfg.local_rank)
    if cfg.ema:
        if cfg.local_rank == 0:
            print("Initializing EMA model..")
        ema_model = ModelEMA(
            model, 
            decay=cfg.ema_decay, 
            device=cfg.local_rank,
        )
    else:
        ema_model = None
    
       # === Move model to device ===
    model = model.to(rank)

    # === EMA (optional) ===
    ema_model = ModelEMA(model, decay=cfg.ema_decay, device=rank) if cfg.ema else None
    if cfg.ema and rank == 0:
        print("Initialized EMA model.")

    # === Wrap in DDP if needed ===
    if is_ddp:
        model = DDP(model, device_ids=[rank])


    
    #criterion = nn.L1Loss()
    criterion = nn.SmoothL1Loss(beta=50.0) # improvement #2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs) # Improvement # 4

    scaler = GradScaler()


    # ========== Training ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Give me warp {}, Mr. Mamun.".format(cfg.world_size))
        print("="*25)
    
    best_loss= 1_000_000
    val_loss= 1_000_000

    for epoch in range(0, cfg.epochs+1):
        if epoch != 0:
            tstart= time.time()
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
    
            # Train loop
            print(f"Training Loop Epoch #:{epoch}")
            model.train()
            total_loss = []
            for i, (x, y) in enumerate(train_dl):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
        
                with autocast():
                    logits = model(x)
                    
                loss = criterion(logits, y)
        
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
        
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
                total_loss.append(loss.item())
                
                if ema_model is not None:
                    ema_model.update(model)
                    
                if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                    train_loss = np.mean(total_loss)
                    total_loss = []
                    print("Epoch {}:     Train MAE: {:.2f}     Val MAE: {:.2f}     Time: {}     Step: {}/{}".format(
                        epoch, 
                        train_loss,
                        val_loss,
                        format_time(time.time() - tstart),
                        i+1, 
                        len(train_dl)+1, 
                    ))
    
        # ========== Valid ==========
        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for x, y in tqdm(valid_dl, disable=cfg.local_rank != 0):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
    
                with autocast():
                    if ema_model is not None:
                        out = ema_model.module(x)
                    else:
                        out = model(x)

                val_logits.append(out.cpu())
                val_targets.append(y.cpu())

            val_logits= torch.cat(val_logits, dim=0)
            val_targets= torch.cat(val_targets, dim=0)
                
            loss = criterion(val_logits, val_targets).item()

        # Gather loss
        v = torch.tensor([loss], device=cfg.local_rank)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
        val_loss = (v[0] / cfg.world_size).item()

        # Improvement # 4 Cosine Annealing Scheduler
        scheduler.step()

        # ========== Weights / Early stopping ==========
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es= cfg.early_stopping
            if val_loss < best_loss:
                print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                print("Saved weights..")
                best_loss = val_loss
                if ema_model is not None:
                    torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                else:
                    torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
        
                es["streak"] = 0
            else:
                es= cfg.early_stopping
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Ending training (early_stopping).")
                    stop_train = torch.tensor([1], device=cfg.local_rank)
        
        # Exits training on all ranks
       # Only broadcast if running with multiple GPUs and DDP is initialized
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(stop_train, src=0)
        if stop_train.item() == 1:
            return

    return
    


if __name__ == "__main__":

    # GPU Specs
    rank = int(os.environ.get("RANK", 0))  # Default rank 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # Default world size 1
    _, total = torch.cuda.mem_get_info(device=rank)

    is_ddp = setup(rank=rank, world_size=world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Rank: {rank}, World size: {world_size}, GPU memory: {total:.2f}GB", flush=True)

    set_seed(cfg.seed + rank)

    cfg.local_rank = rank
    cfg.world_size = world_size
    main(cfg, is_ddp)

    cleanup()
