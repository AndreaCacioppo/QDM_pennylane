import os
import sys
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

# look outside of current directory for modules
sys.path.append(os.getcwd())

from neural_network import Encoder, Decoder
from hyperparameters import *

# hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
LATENT_DIM = 8

# if Autoencoder/encoder.pth and Autoencoder/decoder.pth exist, print warning and ask user if they want to overwrite them
try:
    with open("Autoencoder/encoder.pth", "r") as f:
        print("WARNING: Autoencoder/encoder.pth already exists")
        overwrite = input("Do you want to overwrite it? (y/n) ")
        if overwrite == "y":
            pass
        else:
            quit()
    with open("Autoencoder/decoder.pth", "r") as f:
        print("WARNING: Autoencoder/decoder.pth already exists")
        overwrite = input("Do you want to overwrite it? (y/n) ")
        if overwrite == "y":
            pass
        else:
            quit()
except FileNotFoundError:
    pass

####### function definitions #######
def train_step(encoder, decoder, train_loader, criterion, optimizer, device):
    encoder.train()
    decoder.train()
    running_loss = 0.0

    for data in tqdm(train_loader, desc="Training"):

        # encode data
        data = data.to(device).unsqueeze(1)
        latent = encoder(data) # encoder produces already positive normalized latent vectors

        # decode latent vectors
        outputs = decoder(latent)
        
        # compute loss and update weights
        optimizer.zero_grad()

        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

#####################################

def eval_step(encoder, decoder, test_loader, criterion, device):
    encoder.eval()
    decoder.eval()
    running_loss = 0.0

    with torch.no_grad():
        for data in tqdm(test_loader, desc = "Evaluating"):

            # Encode data
            data = data.to(device).unsqueeze(1)
            latent = encoder(data.to(device)) # encoder produces already positive normalized latent vectors

            # Decode latent vectors
            outputs = decoder(latent)

            # Compute loss
            loss = criterion(outputs, data)
            running_loss += loss.item()

    return running_loss / len(test_loader)

####### main #######
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using {device}")

# Load dataset 
try:
    dataset = torch.load("Data/mnist.pth")
    labels = torch.load("Data/labels.pth")
except FileNotFoundError: 
    print("Run generate_dataset.py first")
    quit()

# assert that dataset mins are 0s and maxs are 1s with a tolerance of 1e-5
assert torch.all(torch.abs(torch.min(dataset) - torch.tensor(0.0)) < 1e-5)
assert torch.all(torch.abs(torch.max(dataset) - torch.tensor(1.0)) < 1e-5)

# create Data folder if it doesn't exist
os.makedirs('Autoencoder', exist_ok=True)

# Create DataLoaders
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# define autoencoder
encoder = Encoder(LATENT_DIM)
decoder = Decoder(LATENT_DIM)

# define loss function and optimizer
criterion = nn.HuberLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr  = LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = SCHEDULER_PATIENCE, gamma = SCHEDULER_GAMMA , verbose = False)

best_loss = 1e9

# if "Results" folder is not present, create it
os.makedirs('Results', exist_ok=True)

# train autoencoder
for epoch in range(1, NUM_EPOCHS + 1):

    train_loss = train_step(encoder, decoder, train_loader, criterion, optimizer, device)
    test_loss = eval_step(encoder, decoder, test_loader, criterion, device)

    print("======================================================")
    print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")

    scheduler.step()

    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch
        torch.save(encoder.to("cpu").state_dict(), "Autoencoder/encoder.pth")
        torch.save(decoder.to("cpu").state_dict(), "Autoencoder/decoder.pth")
        print("Best model saved!")
        encoder.to(device)
        decoder.to(device)

    print("----------------------------------------------")
