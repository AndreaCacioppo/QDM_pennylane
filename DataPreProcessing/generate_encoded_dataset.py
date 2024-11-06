'''takes MNIST and compresses it using the Encoder, creating a latent dataset'''

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

NUM_DIGITS = 10

# load encoder
encoder = Encoder(LATENT_DIM)
encoder.load_state_dict(torch.load("Autoencoder/encoder.pth", map_location = torch.device('cpu')))
encoder.eval()

# load MNIST dataset and labels
try:
        dataset = torch.load("Data/mnist.pth")
        labels = torch.load("Data/labels.pth")
except FileNotFoundError: 
       print("Run generate_reduced_mnist.py first")
       quit()

latent_dataset = torch.tensor([])
latent_labels = torch.tensor([])

for digit in range(NUM_DIGITS):
        
        # create a list for each digit to store the latent vectors
        latent_vectors = []
        
        # iterate over the dataset and extract the latent vectors
        for i in tqdm(range(len(dataset)), desc=f'Encoding latent vectors for digit {digit}', leave=False):
                if labels[i] == digit: latent_vectors.append(encoder(dataset[i].unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy())

        # count the number of samples for the current digit
        num_samples = len(latent_vectors)

        # add labels to the latent dataset
        latent_labels = torch.cat((latent_labels, torch.tensor([digit]*num_samples)), 0)
        
        # concatenate the latent vectors and append to the latent dataset
        latent_dataset = torch.cat((latent_dataset, torch.tensor(np.array(latent_vectors))), 0)

print("Saving latent dataset and labels...")

# save the latent dataset and labels
torch.save(latent_dataset, "Data/latent_mnist.pth")
torch.save(latent_labels.int(), "Data/latent_labels.pth")
