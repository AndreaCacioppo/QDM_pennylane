'''takes MNIST and generates a dataset having the same shape using the diffusion model'''

import os
import sys
import torch
import shutil
from tqdm import tqdm
import torchvision

# look outside of current directory for modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from hyperparameters import *
from Modules.pennylane_functions import *
from Modules.training_functions import *
from Modules.devices import *
from Modules.plot_functions import *
from Autoencoder.neural_network import *

BEST_EPOCH = 95

# load decoder
decoder = Decoder(LATENT_DIM)
decoder.load_state_dict(torch.load("../Autoencoder/decoder.pth", map_location = torch.device('cpu')))
decoder.eval().to(device)

# open directory to store generated samples
if not os.path.exists('generated_samples'): os.mkdir('generated_samples')

# assert that Data folder exists
assert os.path.exists('../Data'), 'Data folder does not exist. Run generate_dataset.py first.'

# download mnist
dataset = torchvision.datasets.MNIST(root='../Data', download=True, transform=torchvision.transforms.ToTensor())
dataset, labels = dataset.data, dataset.targets

# Determine the number of samples per digit to match MNIST
samples_per_digit = [list(labels).count(i) for i in range(10)]

# Iterate over each digit and generate the required number of samples
with torch.no_grad():
    for digit in range(10):
        # Number of samples for the current digit
        num_samples = samples_per_digit[digit]

        # Initialize tqdm progress bar
        pbar = tqdm(total=num_samples, desc=f'Generating for digit {digit}', leave=False)

        # Generate and denoise samples
        label_batch = torch.tensor([digit] * num_samples, device=device)
        batch = torch.view_as_complex(torch.randn(num_samples, LATENT_DIM, 2, device=device))

        for t in range(T-1, -1, -1):
            # Load params
            params = torch.load(f'../{path}Params/trained_params_timestep{t}_epoch{BEST_EPOCH}.pt', map_location=device)
            S1params = params['S1params']
            Xparams = params['Xparams']
            S2params = params['S2params']

            batch = batch / torch.linalg.vector_norm(batch, dim=1).view(-1, 1)
            # Perform denoising step
            batch = circuit(S1params, Xparams, S2params, batch, label_batch)

        batch = torch.abs(batch).float()
        decoded_samples = decoder(batch)

        # Save generated samples and labels to file
        torch.save((decoded_samples), f'generated_samples/digit_{digit}.pth')

        # Free up memory
        del batch, decoded_samples

        # Update tqdm progress bar
        pbar.update(num_samples)

        # Close tqdm progress bar
        pbar.close()

print('All samples generated successfully. Now loading them into memory...')

# Load generated samples into memory
generated_samples = torch.tensor([], device=device)
generated_labels = torch.tensor([], device=device)

for digit in range(10):
    # Load samples and labels from file
    samples = torch.load(f'generated_samples/digit_{digit}.pth')
    labels = torch.tensor([digit] * samples_per_digit[digit], device=device)

    # Append to list
    generated_samples = torch.cat((generated_samples, samples))
    generated_labels = torch.cat((generated_labels, labels))

# Save generated samples and labels to file
torch.save(generated_samples, '../Data/decoded_diffusion_dataset.pth')
torch.save(generated_labels, '../Data/decoded_diffusion_labels.pth')

# Remove generated_samples directory with its contents
shutil.rmtree('generated_samples')

print('All samples loaded successfully!')
