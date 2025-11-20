# evaluateCNN.py
import sys
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from CNN_VGG11 import VGG11Net, VGG11NetDeeper, VGG11NetKernel5
from runCNN import get_limited_indices_by_class  # reuse helper

# ----------------- MODEL SELECTION BY COMMAND LINE -----------------
# Usage:
#   python3 evaluateCNN.py baseline
#   python3 evaluateCNN.py deeper
#   python3 evaluateCNN.py kernel5

if len(sys.argv) != 2:
    print("\nUsage: python3 evaluateCNN.py [baseline | deeper | kernel5]\n")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
if MODEL_NAME not in ["baseline", "deeper", "kernel5"]:
    print("\nInvalid model name. Choose from: baseline, deeper, kernel5\n")
    sys.exit(1)
# -------------------------------------------------------------------


def get_test_loader(batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    test_full = datasets.CIFAR10(root="./data", train=False,
                                 download=True, transform=transform)
    test_indices = get_limited_indices_by_class(test_full, num_per_class=100)
    test_subset = Subset(test_full, test_indices)

    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return test_loader


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics_from_cm(cm):
    num_classes = cm.shape[0]
    eps = 1e-12

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision[c] = tp / (tp + fp + eps)
        recall[c] = tp / (tp + fn + eps)
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + eps)

    return precision, recall, f1, precision.mean(), recall.mean(), f1.mean()


def build_model(model_name, device):
    if model_name == "baseline":
        return VGG11Net(num_classes=10).to(device)
    elif model_name == "deeper":
        return VGG11NetDeeper(num_classes=10).to(device)
    elif model_name == "kernel5":
        return VGG11NetKernel5(num_classes=10).to(device)


def main():
    print(f"\nEvaluating model CNN variant: {MODEL_NAME}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(MODEL_NAME, device)
    model_path = f"models/cnn_{MODEL_NAME}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = get_test_loader()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    cm = compute_confusion_matrix(all_labels, all_preds)
    precision, recall, f1, mp, mr, mf1 = compute_metrics_from_cm(cm)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {mp:.4f}")
    print(f"Macro Recall:    {mr:.4f}")
    print(f"Macro F1-score:  {mf1:.4f}\n")

    print("Per-class metrics:")
    for c in range(10):
        print(f"Class {c}: P={precision[c]:.4f}, R={recall[c]:.4f}, F1={f1[c]:.4f}")

    print("\nConfusion Matrix (rows = true, cols = pred):")
    print(cm)

    np.save(f"models/cnn_{MODEL_NAME}_confusion.npy", cm)

    with open(f"models/cnn_{MODEL_NAME}_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro Precision: {mp:.4f}\n")
        f.write(f"Macro Recall: {mr:.4f}\n")
        f.write(f"Macro F1: {mf1:.4f}\n")

    print(f"\nSaved confusion matrix and metrics in models/ folder.\n")


if __name__ == "__main__":
    main()
