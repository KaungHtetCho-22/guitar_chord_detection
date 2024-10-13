import torch
import torch.nn as nn
import torch.nn.functional as F

class GuitarChordCNN(nn.Module):
    def __init__(self, num_classes):
        super(GuitarChordCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # We'll calculate the correct size for the fully connected layer
        self.fc1_size = self._get_fc1_size()
        
        self.fc1 = nn.Linear(self.fc1_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_fc1_size(self):
        # Create a dummy input to calculate the size
        x = torch.randn(1, 1, 128, 224)  # Assuming input size is 1x128x224
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel() // x.size(0)  # Total number of elements divided by batch size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# # Test the model
# guitarchordcnn = GuitarChordCNN(8)
# print(guitarchordcnn)

# # Test with a sample input
# sample_input = torch.randn(1, 1, 128, 224)  # Batch size 1, 1 channel, 128x224 image
# output = guitarchordcnn(sample_input)
# print(f"Output shape: {output.shape}")