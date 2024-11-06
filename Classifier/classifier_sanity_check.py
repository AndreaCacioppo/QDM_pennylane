'''check that generate_reduced_mnist.py works properly'''

import os
import sys
import torch
from tqdm import tqdm

# look outside of current directory for modules
sys.path.append(os.getcwd())

from hyperparameters import *
from Modules.pennylane_functions import *
from Modules.training_functions import *
from Modules.devices import *
from Modules.plot_functions import *
from Autoencoder.neural_network import *
from neural_network import Classifier

dataset = torch.load("Data/mnist.pth")
labels = torch.load("Data/labels.pth")

# shuffle dataset and labels
indexes = torch.randperm(len(dataset))
dataset = dataset[indexes]
labels = labels[indexes]

dataset = dataset[:10000]
labels = labels[:10000]

dataset = dataset.unsqueeze(1)

# load classifier
classifier = Classifier()

# load params
loaded_params = torch.load("Classifier/best_model.pth")
classifier.load_state_dict(loaded_params)

# take batches and get predictions
predicted_labels = []
for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    predicted_labels.append(classifier(dataset[i:i+BATCH_SIZE]))

predicted_labels = torch.cat(predicted_labels)

# make labels integers
predicted_labels = torch.argmax(predicted_labels, dim = 1)

# print accuracy
print(f"Accuracy on MNIST: {torch.sum(predicted_labels == labels).item()/len(labels)}")
