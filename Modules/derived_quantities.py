import math
import torch
import numpy as np

from hyperparameters import *
from Modules.devices import *

# define constant quantities for the diffusion model
betas      = np.linspace(beta0, betaT, T+1)
alphas     = 1 - betas
alphas_bar = np.cumprod(alphas)
pi         = math.pi

# move everything to torch device
betas      = torch.tensor(betas).float().to(device)
alphas     = torch.tensor(alphas).float().to(device)
alphas_bar = torch.tensor(alphas_bar).float().to(device)
