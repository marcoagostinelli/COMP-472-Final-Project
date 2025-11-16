import torch
import numpy as np
from MLP import (MLP, MLPDeep, MLPShallow, MLPSmallHidden, MLPLargeHidden,
                 predict_mlp, create_data_loaders)
from evaluateNaiveBayes import evaluate


def evaluate_saved_models():
    """
    Load and evaluate all saved MLP models
    """
    # Load the test dataset
    print("Loading test dataset...")
    X_train = np.load('./processed_data/X_train_pca.npy')
    y_train = np.load('./processed_data/Y_train.npy')
    X_test = np.load('./processed_data/X_test_pca.npy')
    y_test = np.load('./processed_data/Y_test.npy')
    
    # Create data loaders
    _, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Define models to evaluate
    models_config = [
        ("MLP (Base)", MLP(), "mlp_base.pth"),
        ("MLP Deep (5 layers)", MLPDeep(), "mlp_deep.pth"),
        ("MLP Shallow (2 layers)", MLPShallow(), "mlp_shallow.pth"),
        ("MLP Small Hidden (256)", MLPSmallHidden(), "mlp_small_hidden.pth"),
        ("MLP Large Hidden (1024)", MLPLargeHidden(), "mlp_large_hidden.pth"),
    ]
    
    results = {}
    
    for model_name, model, model_path in models_config:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load the saved model
        model.load_state_dict(torch.load(f'./models/{model_path}'))
        model.eval()
        
        # Make predictions
        y_pred = predict_mlp(model, test_loader, device)
        
        # Evaluate
        metrics = evaluate(y_test, y_pred)
        results[model_name] = metrics
        
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
    evaluate_saved_models()
