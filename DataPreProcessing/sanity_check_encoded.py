'''checks that genereate_encoded_data.py works correctly'''

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

NUM_DIGITS = 10

# load decoder
decoder = Decoder(LATENT_DIM)
decoder.load_state_dict(torch.load("Autoencoder/decoder.pth", map_location = torch.device('cpu')))
decoder.eval()

dataset = torch.load("Data/latent_mnist.pth")
labels = torch.load("Data/latent_labels.pth")

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

# decode images
decoded_batch = decoder(samples_batch)

# plot 20 images on two rows, displaying each digit and the corresponding label
fig, ax = plt.subplots(2, 10)
for i in range(2):
    for j in range(10):
        ax[i, j].imshow(decoded_batch[i*10+j].detach().numpy().reshape(28, 28), cmap='gray')
        ax[i, j].set_title(f"{label_batch[i*10+j]}")
        ax[i, j].axis('off')

plt.suptitle("Decoded images")
plt.show()
