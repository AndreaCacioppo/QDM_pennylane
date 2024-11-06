import os
import sys
import torch 
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from hyperparameters import *
from Modules.devices import device
from Modules.training_functions import *
from Modules.pennylane_functions import *
from Autoencoder.neural_network import Decoder

# load decoder
decoder = Decoder(LATENT_DIM)
decoder.load_state_dict(torch.load("../Autoencoder/decoder.pth", map_location = torch.device('cpu')))
decoder.eval()

def show_alphas(dataset, alphas_bar, betas, path):

    # get random index for mnist images
    idx = torch.randint(0, dataset.shape[0], size = (1, ), device=device)

    image1 = dataset[idx]
    images1 = [image1]
    image2 = dataset[idx]
    images2 = [image2]

    # append images using alphas bar and noise step
    temp = image2

    T = len(alphas_bar)
    for t in range(T):
        # use assemble input
        images1.append(torch.abs(assemble_input(image1, [t], alphas_bar)))

        # use noise step
        temp = noise_step(temp, [t], betas)
        images2.append(torch.abs(temp))

    # make a subplot of images and histograms
    fig_alphas, axs = plt.subplots(4, T+1, figsize=(20, 9))

    # set title for the whole figure
    plt.suptitle('Denoising performed with assemble_input() (top) and noise_step() (bottom)', fontsize=16)
    bins = np.arange(-0.5, 2.5, 0.2)
    for i in range(T+1):

        # plot image
        axs[0,i].imshow(images1[i].view(2, 4).cpu().detach().numpy(), cmap='gray')
        axs[0,i].axis('off')
        axs[0,i].set_title('{}'.format(i))

        # calculate histogram
        pixel_values = images1[i].cpu().detach().numpy().flatten()
        histogram, _ = np.histogram(pixel_values, bins=bins)

        # plot histogram
        axs[1,i].bar(bins[:-1], histogram, width = 0.2, color='#0504aa', alpha=0.7)
        axs[1,i].set_xlim([-1, 3])

        # plot image from denoising step
        axs[2,i].imshow(images2[i].view(2, 4).cpu().detach().numpy(), cmap='gray')
        axs[2,i].axis('off')

        # calculate histogram
        pixel_values = images2[i].cpu().detach().numpy().flatten()
        histogram, _ = np.histogram(pixel_values, bins=bins)

        # plot histogram
        axs[3,i].bar(bins[:-1], histogram, width = 0.2, color='#0504aa', alpha=0.7)
        axs[3,i].set_xlim([-1, 3])

        # use alphas_bar[-1].cpu().detach().numpy() as title
        if i == T: axs[2,i//2].set_title('t = {}, alphas_bar[T] = {}'.format(i, alphas_bar[-1].cpu().detach().numpy()))

    # remove ticks
    for ax in axs.flatten(): ax.tick_params(axis='both', which='both', length=0)

    # save figure
    fig_alphas.savefig(path+'Logs/alphas.png')
    plt.close(fig_alphas)

def save_plot(S1params, Xparams, S2params, epoch, path):

    # extract 9 random gaussian (complex) noise images and normalise
    batch = torch.view_as_complex(torch.randn(9, 2**(NUM_QUBITS-4), 2, device = device))

    # take 9 labels
    labels_batch = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # implement denoising loop
    for t in range(T-1, -1, -1):

        # load params
        params = torch.load(str(path)+'Params/trained_params_timestep'+str(t)+'_epoch'+str(epoch)+'.pt', map_location = device)
        S1params = params['S1params']
        Xparams = params['Xparams']
        S2params = params['S2params']

        # normalise
        batch = batch/torch.linalg.vector_norm(batch, dim = 1).view(-1, 1)

        # perform denoising step
        batch = circuit(S1params, Xparams, S2params, batch, labels_batch)
    
    # take absolute value (for displaying image only)
    batch = torch.abs(batch)

    # cast to float 32
    batch = batch.type(torch.float32)

    # decode denoised samples
    decoded_samples = decoder(batch)

    # make a subplot of the denoised images
    fig_denoising, axs = plt.subplots(3, 3, figsize = (10, 6))
    for i in range(3):
        for j in range(3):
            axs[i, j].imshow(decoded_samples[i*3 + j].cpu().detach().numpy().reshape(28, 28), cmap = 'gray')
            axs[i, j].axis('off')
            plt.subplots_adjust(wspace = 0.1, hspace = 0.1)

    # save the denoised images
    fig_denoising.savefig(str(path)+'Images/denoised_images'+str(epoch)+'.png')
    plt.close(fig_denoising)

def draw_circuit(path, S1params, Xparams, S2params):

    with open(path+'Logs/config.txt', 'a+') as f:
        f.write(qml.draw(Sblock, expansion_strategy="device")(S1params.cpu().detach().numpy()))
        f.write("\nAdding ancilla: |\u03C8> --> |0>|\u03C8>\n")
        f.write(qml.draw(Lblock, expansion_strategy="device")(Xparams.cpu().detach().numpy()))
        f.write("\nPost selecting: |0>|\u03C8>+|1>|\u03C6> --> |\u03C8>\n")
        f.write(qml.draw(Sblock, expansion_strategy="device")(S2params.cpu().detach().numpy()))

# Helper function to plot a history
def plot_history(axs, num_rows, num_cols, hystory, row_offset):
    for i in range(num_rows // 4):
        for j in range(num_cols):
            index = i * num_cols + j

            # Check if the index is within the range of hystory
            if index < len(hystory):
                axs[row_offset + i, j].imshow(hystory[index].reshape(2, 2), cmap='gray')
                axs[row_offset + i, j].set_xticks([])
                axs[row_offset + i, j].set_yticks([])
            else:
                # Remove the axis if there's no corresponding image in hystory
                axs[row_offset + i, j].axis('off')
