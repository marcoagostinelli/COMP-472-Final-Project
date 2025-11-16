import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class MLP(nn.Module):
    """
    Three-layer Multi-Layer Perceptron for CIFAR-10 classification
    Architecture:
    - Linear(50, 512) - ReLU
    - Linear(512, 512) - BatchNorm(512) - ReLU
    - Linear(512, 10)
    """
    def __init__(self):
        super(MLP, self).__init__()
        
        # Layer 1: Input to first hidden layer
        self.fc1 = nn.Linear(50, 512)
        self.relu1 = nn.ReLU()
        
        # Layer 2: First hidden to second hidden layer
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Second hidden to output layer
        self.fc3 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x


class MLPDeep(nn.Module):
    """
    Deeper variant of MLP with 5 layers
    For depth experimentation
    """
    def __init__(self):
        super(MLPDeep, self).__init__()
        
        self.fc1 = nn.Linear(50, 512)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        
        # Additional layers
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.fc5(x)
        return x


class MLPShallow(nn.Module):
    """
    Shallower variant of MLP with 2 layers
    For depth experimentation
    """
    def __init__(self):
        super(MLPShallow, self).__init__()
        
        self.fc1 = nn.Linear(50, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        return x


class MLPSmallHidden(nn.Module):
    """
    MLP with smaller hidden layer size (256)
    For hidden layer size experimentation
    """
    def __init__(self):
        super(MLPSmallHidden, self).__init__()
        
        self.fc1 = nn.Linear(50, 256)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x


class MLPLargeHidden(nn.Module):
    """
    MLP with larger hidden layer size (1024)
    For hidden layer size experimentation
    """
    def __init__(self):
        super(MLPLargeHidden, self).__init__()
        
        self.fc1 = nn.Linear(50, 1024)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x


def train_mlp(model, train_loader, criterion, optimizer, device, epochs=50):
    """
    Train the MLP model
    
    Args:
        model: The MLP model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda or cpu)
        epochs: Number of training epochs
    
    Returns:
        Trained model
    """
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return model


def predict_mlp(model, test_loader, device):
    """
    Make predictions using the trained MLP model
    
    Args:
        model: Trained MLP model
        test_loader: DataLoader for test data
        device: Device to run on (cuda or cpu)
    
    Returns:
        numpy array of predictions
    """
    model.to(device)
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    return np.array(all_predictions)


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    """
    Create PyTorch DataLoaders from numpy arrays
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        batch_size: Batch size for training
    
    Returns:
        train_loader, test_loader
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
