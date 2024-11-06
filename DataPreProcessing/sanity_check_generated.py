'''check that generate_reduced_mnist.py works properly'''

import torch
import matplotlib as plt
import sys
import os

# look outside of current directory for modules
sys.path.append(os.getcwd())

from hyperparameters import *
from Modules.pennylane_functions import *
from Modules.training_functions import *
from Modules.devices import *
from Modules.plot_functions import *
from Autoencoder.neural_network import *

dataset = torch.load("Data/mnist.pth")
labels = torch.load("Data/labels.pth")

NUM_DIGITS = 10

# take two of each digit
samples = []
label_samples = []

for i in range(NUM_DIGITS):
    indices = torch.where(labels == i)[0]
    samples.append(dataset[indices[0]])
    samples.append(dataset[indices[1]])
    label_samples.append(labels[indices[0]])
    label_samples.append(labels[indices[1]])

samples_batch = torch.stack(samples)
label_batch = torch.stack(label_samples)

# plot 20 images on two rows, displaying each digit and the corresponding label
fig, ax = plt.subplots(2, 10)
for i in range(2):
    for j in range(10):
        ax[i, j].imshow(samples_batch[i*10+j].detach().numpy().reshape(28, 28), cmap='gray')
        ax[i, j].set_title(f"{label_batch[i*10+j]}")
        ax[i, j].axis('off')

plt.suptitle("MNIST dataset quality check")
plt.show()
