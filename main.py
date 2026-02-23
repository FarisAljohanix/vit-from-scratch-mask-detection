from ViT import ViTClassification, Config
from torch import optim, nn
import torch
import kagglehub
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os


def build():
    path = kagglehub.dataset_download("vijaykumar1799/face-mask-detection")

    root_dir = os.path.join(path, "Dataset")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(root=root_dir, transform=transform)

    train_len = int(len(dataset) * 0.7)
    val_len = int(len(dataset) * 0.2)
    test_len = len(dataset) - (train_len + val_len)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    config = Config()
    model = ViTClassification(config)

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion, train_loader, val_loader, test_loader, dataset, config


if __name__ == "__main__":
    build()
