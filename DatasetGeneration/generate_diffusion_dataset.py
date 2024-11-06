'''generates a latent dataset using the trained diffusion model
having the same shape as the MNIST dataset.
Use "generate_decoded_dataset.py" to generate in the pixel space directly'''

import gc
import os
import sys
import torch
from tqdm import tqdm

# look outside of current directory for modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from hyperparameters import *
from Modules.pennylane_functions import *
from Modules.training_functions import *
from Modules.devices import *
from Modules.plot_functions import *
from Autoencoder.neural_network import *

NUM_DIGITS = 10
BATCH_SIZE = 4096
BEST_EPOCH = 10

# Load MNIST dataset and labels to match the shape
try: labels = torch.load("../Data/labels.pth")
except FileNotFoundError: 
    print("Run generate_dataset.py first")
    quit()

# Determine the number of samples per digit to match MNIST
samples_per_digit = [list(labels).count(i) for i in range(NUM_DIGITS)]

# untrack all gradients
torch.set_grad_enabled(False)

# Iterate over each digit and generate the required number of samples
for digit in range(NUM_DIGITS):

    labels_batch = torch.tensor([digit]*BATCH_SIZE).to(device)

    # create tensor to store latent vectors
    latent_vectors = torch.tensor([], device = device)

    # Number of samples for the current digit
    num_samples = samples_per_digit[digit]

    # Calculate the number of batches and the leftover
    num_batches = num_samples // BATCH_SIZE
    leftover = num_samples % BATCH_SIZE

    # Initialize tqdm progress bar
    pbar = tqdm(total=num_samples, desc=f'Generating for digit {digit}', leave=False)

    for _ in range(num_batches):
        batch = torch.view_as_complex(torch.randn(BATCH_SIZE, LATENT_DIM, 2, device = device))

        for t in range(T-1, -1, -1):

            # load params
            params = torch.load('../'+str(path)+'Params/trained_params_timestep'+str(t)+'_epoch'+str(BEST_EPOCH)+'.pt', map_location = device)
            S1params = params['S1params']
            Xparams = params['Xparams']
            S2params = params['S2params']

            batch = batch/torch.linalg.vector_norm(batch, dim=1).view(-1, 1)
            
            # Perform denoising step
            batch = circuit(S1params, Xparams, S2params, batch, labels_batch)

        # take absolute value parts
        batch = torch.abs(batch).float()

        # concatenate batch to latent_vectors[digit]
        latent_vectors = torch.cat((latent_vectors, batch), dim = 0)

        # Free up memory
        del batch
        gc.collect()

        # Update tqdm progress bar
        pbar.update(BATCH_SIZE)

    # If there is a leftover, process it separately
    if leftover > 0:
        batch = torch.view_as_complex(torch.randn(leftover, LATENT_DIM, 2, device=device))
        labels_batch = torch.tensor([digit]*leftover).to(device)

        for t in range(T-1, -1, -1):

            # load params
            params = torch.load('../'+str(path)+'Params/trained_params_timestep'+str(t)+'_epoch'+str(BEST_EPOCH)+'.pt', map_location = device)
            S1params = params['S1params']
            Xparams = params['Xparams']
            S2params = params['S2params']

            batch = batch/torch.linalg.vector_norm(batch, dim=1).view(-1, 1)
            
            # Perform denoising step
            batch = circuit(S1params, Xparams, S2params, batch, labels_batch)

        # take absolute value parts
        batch = torch.abs(batch).float()

        # concatenate batch to latent_vectors[digit]
        latent_vectors = torch.cat((latent_vectors, batch), dim = 0)

        # Free up memory
        del batch
        gc.collect()

        # Update tqdm progress bar
        pbar.update(leftover)

    # Close tqdm progress bar
    pbar.close()

    # save current sample
    torch.save(latent_vectors, f'../Data/latent_diffusion_digit_{digit}.pth')

    # Free up all memory
    del latent_vectors
    gc.collect()

# load digits
latent_vectors = [torch.load(f'../Data/latent_diffusion_digit_{i}.pth') for i in range(NUM_DIGITS)]

# join dataset
latent_dataset = torch.cat(latent_vectors, dim = 0)

# join labels
latent_labels  = torch.cat([torch.tensor([i]*samples_per_digit[i]) for i in range(NUM_DIGITS)])

# delete individual digits
for i in range(NUM_DIGITS): os.remove(f'../Data/latent_diffusion_digit_{i}.pth')

# save latent dataset and labels
torch.save(latent_dataset, '../Data/latent_diffusion_dataset.pth')
torch.save(latent_labels, '../Data/latent_diffusion_labels.pth')
