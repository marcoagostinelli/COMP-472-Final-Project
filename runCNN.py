# runCNN.py
import ssl
import certifi
ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(
    cafile=certifi.where()
)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from CNN_VGG11 import VGG11Net, VGG11NetDeeper, VGG11NetKernel5

# ----------------- CHOOSE MODEL HERE -----------------
# "baseline", "deeper", or "kernel5"
MODEL_NAME = "kernel5"
# -----------------------------------------------------


def get_limited_indices_by_class(dataset, num_per_class):
    targets = np.array(dataset.targets)
    indices = []

    for c in range(10):
        class_indices = np.where(targets == c)[0]
        selected = class_indices[:num_per_class]
        indices.extend(selected.tolist())

    return indices


def get_cifar10_loaders(batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_full = datasets.CIFAR10(root="./data", train=True,
                                  download=True, transform=transform)
    test_full = datasets.CIFAR10(root="./data", train=False,
                                 download=True, transform=transform)

    train_indices = get_limited_indices_by_class(train_full, num_per_class=500)
    test_indices = get_limited_indices_by_class(test_full, num_per_class=100)

    train_subset = Subset(train_full, train_indices)
    test_subset = Subset(test_full, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total
    acc = correct / total
    return loss, acc


def build_model(model_name, device):
    if model_name == "baseline":
        model = VGG11Net(num_classes=10)
    elif model_name == "deeper":
        model = VGG11NetDeeper(num_classes=10)
    elif model_name == "kernel5":
        model = VGG11NetKernel5(num_classes=10)
    else:
        raise ValueError(f"Unknown MODEL_NAME: {model_name}")

    return model.to(device)


def main():
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training model variant: {MODEL_NAME}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    model = build_model(MODEL_NAME, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs("models", exist_ok=True)
            path = f"models/cnn_{MODEL_NAME}_best.pth"
            torch.save(model.state_dict(), path)
            print(f"  -> Saved new best model to {path} (acc={best_test_acc:.4f})")

    os.makedirs("models", exist_ok=True)
    final_path = f"models/cnn_{MODEL_NAME}_last.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Final model saved as {final_path}")


if __name__ == "__main__":
    main()
