'''trains quantum diffusion model on the latent space of the MNIST dataset'''

import os
import time
import torch
import numpy as np
from torch.autograd import Variable
from IPython.display import clear_output

from Modules.training_functions import *
from Modules.pennylane_functions import *
from Modules.derived_quantities import *
from Modules.plot_functions import *
from Modules.devices import *
from Modules.quality_functions import *
from hyperparameters import *

NUM_QUBITS += 4
criterion = "accuracy" # "quality" or "loss" or "accuracy"
path = "results/"

# open necessary folders
if not os.path.exists(path): os.makedirs(path)
if not os.path.exists(path + 'Images/'): os.makedirs(path + 'Images/')
if not os.path.exists(path + 'Params/'): os.makedirs(path + 'Params/')
if not os.path.exists(path + 'Logs/'): os.makedirs(path + 'Logs/')

# write hyperparameters to file
with open(path + 'Logs/hyperparameters.txt', 'w') as f:
    f.write('NUM_QUBITS = {}\n'.format(NUM_QUBITS))
    f.write('NUM_LARGE = {}\n'.format(NUM_LARGE))
    f.write('NUM_SMAL1 = {}\n'.format(NUM_SMALL1))
    f.write('NUM_SMALL2 = {}\n'.format(NUM_SMALL2))
    f.write('NUM_ANCILLAS = {}\n'.format(1))
    f.write('LEARNING_RATE = {}\n'.format(LEARNING_RATE))
    f.write('BATCH_SIZE = {}\n'.format(BATCH_SIZE))
    f.write('NUM_EPOCHS = {}\n'.format(NUM_EPOCHS))
    f.write('SCHEDULER_PATIENCE = {}\n'.format(SCHEDULER_PATIENCE))
    f.write('SCHEDULER_GAMMA = {}\n'.format(SCHEDULER_GAMMA))
    f.write('T = {}\n'.format(T))
    f.write('betas = {}\n'.format(np.linspace(beta0, betaT, T)))

# save device on file
with open(path + 'Logs/device.txt', 'a') as f: f.write('device = {}'.format(device))

# load dataset and flatten images
dataset = torch.load('Data/latent_mnist.pth').to(device)
labels = torch.load('Data/latent_labels.pth').to(device)

# assert that vectors are positive and with norm 1 within a tolerance
assert torch.all(torch.abs(torch.linalg.norm(dataset, dim = 1) - torch.ones(len(dataset), device = device)) < 1e-5)

# shuffle dataset and labels in the same way and take subset
indices = torch.randperm(len(dataset))
dataset = dataset[indices][:DATA_LENGTH, :]
labels = labels[indices][:DATA_LENGTH]

# saves forward process for inspection
show_alphas(dataset, alphas_bar, betas, path)

# initialise parameters, circuit and optimizer
S1params  = Variable(torch.rand(NUM_SMALL1, NUM_QUBITS, 3)*2*pi, requires_grad=True)
S2params  = Variable(torch.rand(NUM_SMALL2, NUM_QUBITS, 3)*2*pi, requires_grad=True)
Xparams = Variable(torch.rand(NUM_LARGE, NUM_QUBITS+1, 3)*2*pi, requires_grad=True)

optimizer = torch.optim.Adam([S1params, S2params, Xparams], lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = SCHEDULER_PATIENCE, gamma = SCHEDULER_GAMMA, verbose = False)

# make dataloader
dataset = torch.utils.data.TensorDataset(dataset, labels)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

loss_histories = np.empty((0, T))
loss_histories = np.vstack((loss_histories, [1.]*T))
loss_history = []
quality_history = [0.]
accuracy_history = [0.]
best_quality = 0.
best_loss = 1.
best_accuracy = 0.

# save configuration of circuit on file
draw_circuit(path, S1params, Xparams, S2params)

# training loop
for epoch in range(NUM_EPOCHS):

    print("-----------------------------------------")
    
    clear_output(wait=True)

    # Training progress bar
    pbar_train = tqdm(total=len(data_loader), desc='Epoch '+str(epoch+1)+'/'+str(NUM_EPOCHS), leave=False, position=0, unit="batch")
    
    # take batch of images and labels
    for i, (image_batch, label_batch) in enumerate(data_loader):

        # train each time step on the same batch
        for tstep in range(T):

            t0 = time.time()

            # extract batch of random times and betas
            t = torch.tensor([tstep]*len(image_batch), dtype=torch.long)
            betas_batch = betas[t].to(device)

            # assemble input at t and add noise (t+1)
            target_batch = assemble_input(image_batch, t, alphas_bar) #q(x_t|x_0)
            input_batch  = noise_step(target_batch, t+1, betas) #q(x_t|x_(t-1))

            # get target and input (always real)
            target_batch = target_batch / torch.linalg.vector_norm(target_batch, dim = 1).view(-1, 1)
            input_batch  = input_batch / torch.linalg.vector_norm(input_batch, dim = 1).view(-1, 1)
            
            # Feed to circuit, compute the loss and update the weights
            optimizer.zero_grad()
            loss = loss_fn(S1params, Xparams, S2params, input_batch, target_batch, label_batch)
            loss.backward()
            optimizer.step()

            torch.save({'S1params': S1params, 'S2params': S2params, 'Xparams': Xparams}, str(path)+'Params/trained_params_timestep'+str(tstep)+'_epoch'+str(epoch)+'.pt')
        
            # append losses
            loss_history.append(loss.item())
        
        # Update training progress bar
        pbar_train.update(1)

        # delete unless last batch
        if i != len(data_loader)-1: del loss_history[:]
            
    pbar_train.close()

    # turn loss history into numpy vector
    loss_histories = np.vstack((loss_histories, loss_history))

    # Transpose the data for plotting
    transposed_data = loss_histories.T

    # Plot each line
    for i, line in enumerate(transposed_data): plt.plot(line, label=f'Denoising {i+1} --> {i}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path + f'Logs/loss_plot.png')
    plt.close()

    # evaluate quality
    quality, accuracy = quality_eval(epoch, path)

    # append parameters and print loss
    loss_history.append(loss.item())
    quality_history.append(quality.item())
    accuracy_history.append(accuracy.item()/100.)

    # implement learning rate scheduler
    scheduler.step()

    # print on file and save loss history plot
    with open(path + 'Logs/loss.txt', 'a+') as f:
        f.write('Epoch: {}/{} - Loss: {:.3f} - Quality: {:.3f} - Time: {:.2f}s\n'.format(epoch+1, NUM_EPOCHS, loss.item(), quality.item(), time.time()-t0))
        plt.clf()  # Clear the previous plot
        plt.plot(quality_history, label = 'Quality', color = 'blue')
        plt.plot(accuracy_history, label = 'Accuracy', color = 'green')
        if epoch == 0: plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path + 'Logs/quality_plot.png')
        plt.close()

    print('Epoch {}/{} - Loss {:.3f} - Quality {:.3f} - Accuracy {:.1f}%'.format(epoch+1, NUM_EPOCHS, loss.item(), quality, accuracy))

    # save every 5 epochs
    if epoch % 5 == 0: save_plot(S1params, Xparams, S2params, epoch, path)
