import torch

from Modules.pennylane_functions import *
from Modules.derived_quantities import *
from Modules.devices import *

# perfrom t noise steps (formula (4) in the DDPM paper): x0 --> xt
def assemble_input(x0, t, alphas_bar):

    # compute batch of coefficients
    alphas_bar_batch = alphas_bar[t]

    # compute averages
    avgs = torch.sqrt(alphas_bar_batch).unsqueeze(1).repeat(1, x0.shape[1])*x0

    # compute fluctuations
    fluct = (1-alphas_bar_batch).unsqueeze(1).repeat(1, x0.shape[1])*torch.view_as_complex(torch.randn(x0.shape[0], x0.shape[1], 2, device = device))

    # perform t noise steps
    noisy_image = avgs+fluct

    return noisy_image

# perfrom a noise step (formula (2) in the DDPM paper): xt-1 --> xt
def noise_step(xt_1, t, betas):
        
        # compute batch of coefficients
        betas_batch = betas[t]

        # compute averages by rescaling the input image
        avgs = torch.sqrt(1-betas_batch).unsqueeze(1).repeat(1, xt_1.shape[1])*xt_1

        # get fluctuations by rescaling Gaussian complex noise
        fluct = betas[t].unsqueeze(1).repeat(1, xt_1.shape[1])*torch.view_as_complex(torch.randn(xt_1.shape[0], xt_1.shape[1], 2, device = device))

        # perform noise step
        x_t = avgs+fluct

        return x_t

# Define the cost function
def loss_fn(S1params, Xparams, S2params, input_batch, target_batch, labels_batch):

    # Compute the output states
    if Xparams.shape[1] == S1params.shape[1]+1: output_batch = circuit(S1params, Xparams, S2params, input_batch, labels_batch)
    elif Xparams.shape[1] == S1params.shape[1]-1: output_batch = auto_encoder(S1params, Xparams, S2params, input_batch, labels_batch)
    
    # compute the fidelity between the output states and the target states
    fid = torch.abs(torch.sum(torch.conj(output_batch)*target_batch, dim=1))**2

    return 1-fid.mean()
