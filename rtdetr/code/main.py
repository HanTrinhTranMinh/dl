import os
import random
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dummy_dataset import DummyDetectionDataset, collate_fn
from models.presnet import build_presnet
from models.hybrid_encoder import HybridEncoder
from models.rtdetr_head import build_rtdetr_head
from models.transformer_decoder import build_transformer_decoder
from models.hungarian_matcher import build_hungarian_matcher
from models.detr_loss import build_detr_loss


# =========================================================
# Utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_targets_to_device(targets, device):
    new_targets = []
    for t in targets:
        new_targets.append({
            "labels": t["labels"].to(device),
            "boxes": t["boxes"].to(device),
        })
    return new_targets


# =========================================================
# Full Model Wrapper
# =========================================================
class RTDETRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_presnet(cfg["model"]["backbone"])

        neck_cfg = cfg["model"]["neck"]
        self.neck = HybridEncoder(
            in_channels=tuple(neck_cfg["in_channels"]),
            hidden_dim=neck_cfg["hidden_dim"],
            num_encoder_layers=neck_cfg["num_encoder_layers"],
            num_heads=neck_cfg["num_heads"],
            ffn_dim=neck_cfg["ffn_dim"],
            csp_blocks=neck_cfg["csp_blocks"],
        )

        self.head = build_rtdetr_head(cfg["model"]["head"])
        self.decoder = build_transformer_decoder(cfg["model"]["decoder"])

    def forward(self, images, targets=None):
        # 1) backbone
        feats = self.backbone(images)   # {"s3":..., "s4":..., "s5":...}

        # 2) neck
        feats = self.neck(feats)        # {"s3":..., "s4":..., "s5":...}

        # 3) head: encoder-side query selection + denoising
        head_out = self.head(feats, targets=targets)

        decoder_inputs = head_out["decoder_inputs"]

        # 4) decoder
        dec_out = self.decoder(
            tgt=decoder_inputs["tgt"],
            ref_points=decoder_inputs["ref_points"],
            memory=decoder_inputs["memory"],
            spatial_shapes=decoder_inputs["spatial_shapes"],
            level_start_index=decoder_inputs["level_start_index"],
            attn_mask=decoder_inputs["attn_mask"],
        )

        # gắn thêm encoder outputs / dn meta nếu cần debug
        dec_out["enc_outputs"] = head_out["enc_outputs"]
        dec_out["dn_meta"] = head_out["dn_meta"]

        return dec_out


# =========================================================
# Optimizer
# =========================================================
def build_optimizer(cfg, model):
    base_lr = cfg["optimizer"]["lr"]
    backbone_lr = cfg["optimizer"].get("backbone_lr", base_lr)
    weight_decay = cfg["optimizer"]["weight_decay"]

    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": base_lr},
        ],
        weight_decay=weight_decay
    )
    return optimizer


# =========================================================
# Train / Val loops
# =========================================================
def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch, print_freq=10):
    model.train()

    running_loss = 0.0

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad()

        outputs = model(images, targets=targets)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss_total"]

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % print_freq == 0:
            msg = f"[Train] Epoch {epoch} Step {step}/{len(dataloader)} Loss: {loss.item():.4f}"
            for k, v in loss_dict.items():
                msg += f" | {k}: {float(v.detach()):.4f}"
            print(msg)

    return running_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate_one_epoch(model, criterion, dataloader, device, epoch, print_freq=10):
    model.eval()

    running_loss = 0.0

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = move_targets_to_device(targets, device)

        outputs = model(images, targets=targets)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss_total"]

        running_loss += loss.item()

        if step % print_freq == 0:
            msg = f"[Val] Epoch {epoch} Step {step}/{len(dataloader)} Loss: {loss.item():.4f}"
            print(msg)

    return running_loss / max(len(dataloader), 1)


# =========================================================
# Main
# =========================================================
def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # -----------------------------
    # Dataset / DataLoader
    # -----------------------------
    train_dataset = DummyDetectionDataset(
        num_samples=cfg["dataset"]["train_size"],
        image_size=cfg["dataset"]["image_size"],
        num_classes=cfg["dataset"]["num_classes"],
        max_objects=cfg["dataset"]["max_objects"],
    )

    val_dataset = DummyDetectionDataset(
        num_samples=cfg["dataset"]["val_size"],
        image_size=cfg["dataset"]["image_size"],
        num_classes=cfg["dataset"]["num_classes"],
        max_objects=cfg["dataset"]["max_objects"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = RTDETRModel(cfg).to(device)

    # -----------------------------
    # Matcher + Criterion
    # -----------------------------
    matcher = build_hungarian_matcher(cfg["matcher"])
    criterion = build_detr_loss(cfg["loss"], matcher).to(device)

    # -----------------------------
    # Optimizer + Scheduler
    # -----------------------------
    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler"]["step_size"],
        gamma=cfg["scheduler"]["gamma"]
    )

    # -----------------------------
    # Train loop
    # -----------------------------
    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=cfg["print_freq"]
        )

        val_loss = validate_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            print_freq=cfg["print_freq"]
        )

        scheduler.step()

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}")
        print(f"  LR: {[group['lr'] for group in optimizer.param_groups]}\n")

        # save last
        last_path = os.path.join(cfg["save_dir"], "last.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
        }, last_path)

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg["save_dir"], "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, best_path)
            print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/rtdetrv2_custom.yaml")
    args = parser.parse_args()

    main(args)