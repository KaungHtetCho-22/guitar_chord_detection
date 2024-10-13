
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
from model import GuitarChordCNN

import torch.nn.functional as F



class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir, target_width=224):  # Define a fixed target width
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.label_map = {chord: idx for idx, chord in enumerate(set([f.split('_')[0] for f in self.file_list]))}
        self.target_width = target_width  # Set a target width for resizing

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        mel_spec = np.load(file_path)
        mel_spec = torch.from_numpy(mel_spec).unsqueeze(0).float()

        # Resize the mel-spectrogram to the target width
        mel_spec = F.interpolate(mel_spec.unsqueeze(0), size=(mel_spec.size(1), self.target_width), mode='bilinear', align_corners=False)
        mel_spec = mel_spec.squeeze(0)

        label = self.label_map[self.file_list[idx].split('_')[0]]
        return mel_spec, label


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = MelSpectrogramDataset('data/processed/train')
    print(f'train data: {len(train_data)}')
    val_data = MelSpectrogramDataset('data/processed/test')
    print(f'test data: {len(val_data)}')
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    print(f'train dataloader: {len(train_loader)}')

    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    print(f'test loader: {len(val_loader)}')

    
    num_classes = len(train_data.label_map)
    print(f'num classes: {num_classes}')
    model = GuitarChordCNN(num_classes).to(device)
    
    train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, device=device)
    
    torch.save(model.state_dict(), 'weights/guitar_chord_cnn.pth')

if __name__ == "__main__":
    main()