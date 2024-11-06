'''trains 10 classifiers that output the probability that 
the image is 0, ..., 9'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm  # Import tqdm for progress bars

from neural_network import ROC_classifier

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# check if trained models already exist and if so, ask if they should be overwritten
if os.path.exists("ComputeMetrics/trained_models/best_model0.pth") or os.path.exists("ComputeMetrics/trained_models/best_model1.pth"):
    print("WARNING: trained_models already exist")
    overwrite = input("Do you want to overwrite them? (y/n) ")
    if overwrite == "y":
        pass
    else:
        quit()

# Hyperparameters
NUM_DIGITS = 10
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10

# Load MNIST dataset and normalize pixel values
X = torch.load('Data/mnist.pth').to(device)
y = torch.load('Data/labels.pth').to(device)

# Creating DataLoaders
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# take a small subset of the training data for faster training
train_dataset, _ = random_split(train_dataset, [10000, len(train_dataset) - 10000])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Train function
def train(model, train_loader, optimizer, criterion, digit):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels = batch

        inputs = inputs.unsqueeze(1)

        # target vector with ones only where digit == labels using torch.where
        targets = torch.where(labels == digit, torch.ones_like(labels), torch.zeros_like(labels)).float()

        outputs = model(inputs).squeeze()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# Test function
def test(model, test_loader, criterion, digit):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch

            inputs = inputs.unsqueeze(1)

            # target vector with ones only where digit == labels
            targets = torch.zeros(len(labels), device = device)

            targets[labels == digit] = 1

            outputs = model(inputs).squeeze()

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(test_loader)

###### Main ######

# Initialize models, optimizers, and criterion for each digit
models = [ROC_classifier() for _ in range(NUM_DIGITS)]
optimizers = [optim.Adam(model.parameters(), lr=LEARNING_RATE) for model in models]
criterion = nn.BCELoss()

# Create a folder for trained models if it doesn't exist
if not os.path.exists("ComputeMetrics/trained_models"):
    os.makedirs("ComputeMetrics/trained_models")

# Training and early stopping for each digit
best_test_losses = [float('inf')] * NUM_DIGITS
patiences = [EARLY_STOPPING_PATIENCE] * NUM_DIGITS

for digit in range(NUM_DIGITS):

    pbar = tqdm(total=NUM_EPOCHS, desc=f'Training digit {digit}', dynamic_ncols=True)

    # Create a tqdm progress bar for training
    for epoch in range(NUM_EPOCHS):

        models[digit].train()

        models[digit] = models[digit].to(device)

        train_loss = train(models[digit], train_loader, optimizers[digit], criterion, digit)
        test_loss = test(models[digit], test_loader, criterion, digit)

        # Check if the test loss has improved
        if test_loss < best_test_losses[digit]:
            best_test_losses[digit] = test_loss
            patiences[digit] = 0  # Reset patience if there's improvement
            torch.save(models[digit].to(torch.device("cpu")).state_dict(), f'ComputeMetrics/trained_models/best_model{digit}.pth')
        else:
            patiences[digit] += 1

            # If test loss hasn't improved for PATIENCE steps, move on to the next digit
            if patiences[digit] >= EARLY_STOPPING_PATIENCE:
                break

        # Compute accuracy for current digit
        models[digit].eval()
        correct = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs = inputs.unsqueeze(1)

                # send model to device
                models[digit] = models[digit].to(device)

                # target vector with ones only where digit == labels
                outputs = models[digit](inputs).squeeze()

                predicted = torch.round(outputs)

                targets = torch.zeros_like(labels)
                targets[labels == digit] = 1

                correct += (predicted == targets).sum().item()

        # Update progress bar description with accuracy
        pbar.set_postfix(Accuracy=f'{100 * correct / len(test_dataset):.2f}%', Patience = patiences[digit])
        pbar.update(1)  # Update the progress bar

    pbar.close()
    print(f'Training for digit {digit} is complete.')
    print("------------------------------------------------------------")

    # If early stopping occurred, move on to the next digit
    if patiences[digit] >= EARLY_STOPPING_PATIENCE:
        continue
