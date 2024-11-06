import torch
from torch import nn

# Define the classifier
class ROC_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 14 * 14, 256) # input size will be half because of the max pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.MaxPool2d(2, 2)(x) # applying max pooling of size 2
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Softmax activation for classification
        return x
