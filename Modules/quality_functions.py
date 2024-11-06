# define function for quality evaluation at each epoch
# which generates N samples for each digit and
# uses a trained classifier to calculate the accuracy

import torch
from tqdm import tqdm

# look outside the current directory for modules
import sys
sys.path.append('../')

from Classifier.neural_network import Classifier
from hyperparameters import *
from Autoencoder.neural_network import *
from Modules.devices import *
from Modules.pennylane_functions import *

NUM_DIGITS = 10

# load classifier
classifier = Classifier()
classifier.load_state_dict(torch.load("Classifier/best_model.pth", map_location = torch.device('cpu')))
classifier.eval()

# load decoder
decoder = Decoder(LATENT_DIM)
decoder.load_state_dict(torch.load("Autoencoder/decoder.pth", map_location = torch.device('cpu')))
decoder.eval()

def quality_eval(epoch, path, N = 20):

    # extract batch of Gaussian noise of shape (NUM_DIGITS*N, 8)
    batch = torch.view_as_complex(torch.randn(NUM_DIGITS*N, LATENT_DIM, 2, device = device))

    # take labels with shape (NUM_DIGITS*N)
    labels_batch = torch.tensor([i for i in range(NUM_DIGITS) for j in range(N)])

    # validation progress bar
    pbar_val = tqdm(total=T, desc='Validation', leave=False)

    # implement denoising loop
    for t in range(T-1, -1, -1):

        params = torch.load(str(path)+'Params/trained_params_timestep'+str(t)+'_epoch'+str(epoch)+'.pt')
        S1params = params['S1params']
        S2params = params['S2params']
        Xparams  = params['Xparams']

        # normalise batch
        batch = batch/torch.linalg.vector_norm(batch, dim = 1).view(-1, 1)
        
        # perform denoising step
        batch = circuit(S1params, Xparams, S2params, batch, labels_batch)

        pbar_val.update()

    pbar_val.close()

    # take real parts and normalize (for displaying image only)
    batch = torch.abs(batch).float()
    batch = batch/torch.linalg.vector_norm(batch, dim = 1).view(-1, 1)

    # decode denoised samples to get images
    decoded_samples = decoder(batch)

    # turn to one-hot encoding
    labels_batch = torch.nn.functional.one_hot(labels_batch, num_classes = NUM_DIGITS)

    # get predictions for decoded samples
    predictions = classifier(decoded_samples)

    # get quality computing MAE between predictions and labels
    loss = torch.mean(torch.abs(predictions - labels_batch))
    
    # compute accuracy
    accuracy = torch.mean((torch.argmax(predictions, dim = 1) == torch.argmax(labels_batch, dim = 1)).float())*100

    return 1-loss, accuracy
