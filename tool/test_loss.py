import torch
import yaml
from torch.utils.data import DataLoader

from main import RTDETRModel
from my_rtdetr.tool.Dummy_dataset import DummyDetectionDataset, collate_fn
from models.hungarian_matcher import build_hungarian_matcher
from models.detr_loss import build_detr_loss

cfg = yaml.safe_load(open("config/resnet18v_d_custom.yaml", "r", encoding="utf-8"))

dataset = DummyDetectionDataset(num_samples=4, image_size=640, num_classes=4, max_objects=5)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

images, targets = next(iter(loader))

model = RTDETRModel(cfg)
model.eval()

matcher = build_hungarian_matcher(cfg["matcher"])
criterion = build_detr_loss(cfg["loss"], matcher)

with torch.no_grad():
    outputs = model(images, targets=targets)
    loss_dict = criterion(outputs, targets)

for k, v in loss_dict.items():
    print(k, float(v.detach()))