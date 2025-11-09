import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import Subset, DataLoader
from sklearn.decomposition import PCA
import numpy as np
import os


# Define transformations for the dataset
# Resize eachn image to 224x224
transform = transforms.Compose([
    transforms.Resize(224),  # required for ResNet feature extraction
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Download the CIFAR-10 dataset with our defined transformations
trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)




# Select the first 500 training and 100 test images per class
train_indices = defaultdict(list)
test_indices = defaultdict(list)

for idx, (_, label) in enumerate(trainset_full):
    if len(train_indices[label]) < 500:
        train_indices[label].append(idx)
for idx, (_, label) in enumerate(testset_full):
    if len(test_indices[label]) < 100:
        test_indices[label].append(idx)

#combine all selected indices
selected_train_indices = [i for indices in train_indices.values() for i in indices]
selected_test_indices = [i for indices in test_indices.values() for i in indices]

trainset = Subset(trainset_full, selected_train_indices)
testset = Subset(testset_full, selected_test_indices)

# Extract ResNet-18 features (512 dimensions) for each image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
resnet.to(device)
resnet.eval()


def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            output = resnet(images).squeeze(-1).squeeze(-1) # (batch, 512, 1, 1) -> (batch, 512)
            features.append(output.cpu())
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

X_train, Y_train = extract_features(trainset)
X_test, Y_test = extract_features(testset)

# Apply PCA to reduce dimensions from 512 to 50
X_train_np = X_train.numpy()
X_test_np = X_test.numpy()

pca = PCA(n_components=50)
pca.fit(X_train_np)
X_train_pca = pca.transform(X_train_np)
X_test_pca = pca.transform(X_test_np)

# save the dataset
os.makedirs('./processed_data', exist_ok=True)

np.save('./processed_data/X_train_pca.npy', X_train_pca)
np.save('./processed_data/Y_train.npy', Y_train.numpy())
np.save('./processed_data/X_test_pca.npy', X_test_pca)
np.save('./processed_data/Y_test.npy', Y_test.numpy())
