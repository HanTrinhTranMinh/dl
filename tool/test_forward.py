import sys
sys.path.append(r"E:\Pj\dl")

import torch
import yaml
from torch.utils.data import DataLoader

from my_rtdetr.rtdetr.code.main import RTDETRModel
from my_rtdetr.tool.Dummy_dataset import DummyDetectionDataset, collate_fn

cfg_path = r"E:\Pj\dl\my_rtdetr\rtdetr\config\resnet18v_d_custom.yaml"

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dataset = DummyDetectionDataset(
    num_samples=4,
    image_size=640,
    num_classes=4,
    max_objects=5
)

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn
)

model = RTDETRModel(cfg)
model.eval()

images, targets = next(iter(loader))

with torch.no_grad():
    outputs = model(images, targets=targets)

print("pred_logits:", outputs["pred_logits"].shape)
print("pred_boxes :", outputs["pred_boxes"].shape)

if "aux_outputs" in outputs:
    print("num aux outputs:", len(outputs["aux_outputs"]))