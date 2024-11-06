'''downloads MNIST and takes only 0 and 1, then stores as reduced_mnist.pth and reduced_labels.pth'''

import os
import sys
import torch
import shutil
from torchvision import datasets, transforms

# look outside of current directory for modules
sys.path.append(os.getcwd())

# Download and load MNIST train and test data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Merge train and test data and labels
full_data = torch.cat([train_dataset.data.float(), test_dataset.data.float()])
full_labels = torch.cat([train_dataset.targets, test_dataset.targets])

# delete data folder with its content
shutil.rmtree('./data')

# Normalize data to range [0, 1]
full_data /= 255.0

# create Data folder if it doesn't exist
os.makedirs('Data', exist_ok=True)

# Save reduced data and labels
torch.save(full_data, 'Data/mnist.pth')
torch.save(full_labels, 'Data/labels.pth')
