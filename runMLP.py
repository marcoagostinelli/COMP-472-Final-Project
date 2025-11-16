import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MLP import (MLP, MLPDeep, MLPShallow, MLPSmallHidden, MLPLargeHidden,
                 train_mlp, predict_mlp, create_data_loaders)
from evaluateNaiveBayes import evaluate
import os


def train_and_save_all_models():
    """
    Train all MLP variants and save them
    """
    # Load the dataset
    print("Loading dataset...")
    X_train = np.load('./processed_data/X_train_pca.npy')
    y_train = np.load('./processed_data/Y_train.npy')
    X_test = np.load('./processed_data/X_test_pca.npy')
    y_test = np.load('./processed_data/Y_test.npy')
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create directory for saving models
    os.makedirs('./models', exist_ok=True)
    
    # Define models to train
    models_config = [
        ("MLP (Base)", MLP(), "mlp_base.pth", 0.01),
        ("MLP Deep (5 layers)", MLPDeep(), "mlp_deep.pth", 0.01),
        ("MLP Shallow (2 layers)", MLPShallow(), "mlp_shallow.pth", 0.01),
        ("MLP Small Hidden (256)", MLPSmallHidden(), "mlp_small_hidden.pth", 0.01),
        ("MLP Large Hidden (1024)", MLPLargeHidden(), "mlp_large_hidden.pth", 0.005),
    ]
    
    results = {}
    
    for model_name, model, save_path, lr in models_config:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        # Train the model
        trained_model = train_mlp(model, train_loader, criterion, optimizer, device, epochs=50)
        
        # Save the model
        torch.save(trained_model.state_dict(), f'./models/{save_path}')
        print(f"Model saved to ./models/{save_path}")
        
        # Evaluate the model
        print("\nEvaluating on test set...")
        y_pred = predict_mlp(trained_model, test_loader, device)
        metrics = evaluate(y_test, y_pred)
        
        results[model_name] = metrics
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL MODELS")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 78)
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")


if __name__ == "__main__":
    train_and_save_all_models()
