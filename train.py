import torch
from tqdm import tqdm

import main


def train_one_epoch(model, optimizer, criterion, data_loader):
    model.train()
    total_loss = 0
    for img, lab in tqdm(data_loader):
        out, _ = model(img)
        loss = criterion(out, lab)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)


def val(model, criterion, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, lab in tqdm(data_loader):
            out, _ = model(img)
            loss = criterion(out, lab)

            total_loss += loss.item()
    return total_loss / len(data_loader)


def train():
    model, optimizer, criterion, train_loader, val_loader, _,dataset, config = main.build()

    for epoch in range(10):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader)
        val_loss = val(model, criterion, val_loader)
        print(f'Train and test losses: {train_loss:.4f}, {val_loss:.4f}')

    checkpoint = {
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "class_to_idx": dataset.class_to_idx,
        "image_size": 224,
    }
    torch.save(checkpoint, "vit.pt")


if __name__ == "__main__":
    train()
