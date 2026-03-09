from torch.utils.data import DataLoader
from Dummy_dataset import DummyDetectionDataset, collate_fn

dataset = DummyDetectionDataset(num_samples=10)

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn
)

for images, targets in loader:
    print("Images:", images.shape)
    print("Targets:", targets)
    break