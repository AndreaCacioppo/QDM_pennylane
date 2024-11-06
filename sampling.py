import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from hyperparameters import *
from Modules.pennylane_functions import *
from Modules.training_functions import *
from Modules.devices import *
from Modules.plot_functions import *
from Autoencoder.neural_network import *

BEST_EPOCH = 7

# load decoder
decoder = Decoder(LATENT_DIM)
decoder.load_state_dict(torch.load("Autoencoder/decoder.pth", map_location = torch.device('cpu')))
decoder.eval()

# extract 9 random gaussian (complex) noise images and normalise
batch = torch.view_as_complex(torch.randn(100, LATENT_DIM, 2, device = device))

# take 4 labels
labels_batch = torch.tensor([0]*10+[1]*10+[2]*10+[3]*10+[4]*10+[5]*10+[6]*10+[7]*10+[8]*10+[9]*10)

images = []

# implement denoising loop
for t in tqdm(range(T-1, -1, -1)):

    # load params
    params = torch.load(str(path)+'Params/trained_params_timestep'+str(t)+'_epoch'+str(BEST_EPOCH)+'.pt', map_location = device)
    S1params = params['S1params']
    Xparams = params['Xparams']
    S2params = params['S2params']

    # normalise
    batch = batch/torch.linalg.vector_norm(batch, dim = 1).view(-1, 1)
    
    # perform denoising step
    batch = circuit(S1params, Xparams, S2params, batch, labels_batch)

# take real parts (for displaying image only)
batch = torch.abs(batch).float()

# decode denoised samples to get images
decoded_samples = decoder(batch)

# make a subplot of the denoised images
fig, axs = plt.subplots(10, 10, figsize = (12, 8))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(decoded_samples[i*10 + j].cpu().detach().numpy().reshape(28, 28), cmap = 'gray')
        axs[i, j].axis('off')
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
plt.show()
