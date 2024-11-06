'''check that classifiers are trained'''

import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

# look outside of current directory for modules
sys.path.append(os.getcwd())

from hyperparameters import *
from Modules.pennylane_functions import *
from Modules.training_functions import *
from Modules.devices import *
from Modules.plot_functions import *
from Autoencoder.neural_network import *
from neural_network import ROC_classifier

NUM_DIGITS = 10

dataset = torch.load("Data/mnist.pth")
labels = torch.load("Data/labels.pth")

# shuffle dataset and labels
indexes = torch.randperm(len(dataset))
dataset = dataset[indexes]
labels = labels[indexes]

dataset = dataset[:10000]
labels = labels[:10000]

test_dataset = TensorDataset(dataset, labels)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize models, optimizers, and criterion for each digit
models = [ROC_classifier() for _ in range(NUM_DIGITS)]

for digit in range(NUM_DIGITS):
    models[digit].load_state_dict(torch.load("ComputeMetrics/trained_models/best_model" + str(digit) + ".pth"))

     # Compute accuracy for current digit
    models[digit].eval()
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.unsqueeze(1)

            # send model to device
            models[digit] = models[digit].to(device)

            # target vector with ones only where digit == labels
            outputs = models[digit](inputs).squeeze()

            predicted = torch.round(outputs)

            targets = torch.zeros_like(labels)
            targets[labels == digit] = 1

            correct += (predicted == targets).sum().item()

    print("Accuracy for digit " + str(digit) + ": " + str(100*correct / len(test_dataset))+ "%")
