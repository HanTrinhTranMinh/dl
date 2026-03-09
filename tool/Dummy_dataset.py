import random
import torch
from torch.utils.data import Dataset


class DummyDetectionDataset(Dataset):
    def __init__(self, num_samples=100, image_size=640, num_classes=4, max_objects=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)

        num_objs = random.randint(1, self.max_objects)
        labels = torch.randint(0, self.num_classes, (num_objs,), dtype=torch.int64)

        # normalized cxcywh
        boxes = torch.rand(num_objs, 4)
        boxes[:, 2:] = boxes[:, 2:] * 0.4
        boxes[:, :2] = boxes[:, :2].clamp(0.1, 0.9)
        boxes[:, 2:] = boxes[:, 2:].clamp(0.05, 0.5)

        target = {
            "labels": labels,
            "boxes": boxes
        }
        return image, target


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets

if __name__ == "__main__":
    dataset = DummyDetectionDataset(num_samples=5)

    print("Dataset length:", len(dataset))

    image, target = dataset[0]

    print("Image shape:", image.shape)
    print("Labels:", target["labels"])
    print("Boxes:", target["boxes"])