# src/models/evaluate.py

import torch
from torch.utils.data import DataLoader
from model import GuitarChordCNN
from train import MelSpectrogramDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('weights/confusion_matrix.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the test data
    test_data = MelSpectrogramDataset('data/processed/test')
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Load the trained model
    num_classes = len(test_data.label_map)
    model = GuitarChordCNN(num_classes).to(device)
    model.load_state_dict(torch.load('weights/guitar_chord_cnn.pth'))
    
    # Evaluate the model
    predictions, true_labels = evaluate_model(model, test_loader, device)
    
    # Generate classification report
    class_names = list(test_data.label_map.keys())
    report = classification_report(true_labels, predictions, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, class_names)
    
    print("Evaluation complete. Confusion matrix saved as 'weights/confusion_matrix.png'")

if __name__ == "__main__":
    main()