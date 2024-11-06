import torch
import torch.nn as nn
import torch.optim as optim

from neural_network import Classifier

# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

# Define the test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def main():
    # Set the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the batch size and number of epochs
    batch_size = 64
    epochs = 100

    dataset = torch.load("Data/mnist.pth")
    labels = torch.load("Data/labels.pth")

    dataset = dataset.unsqueeze(1)

    # Assert that the dataset min and max are 0 and 1 within a tolerance of 1e-5
    assert (dataset.data.min() >= 0 - 1e-5) and (dataset.data.max() <= 1 + 1e-5)

    # split dataset into training and test sets
    train_dataset = torch.utils.data.TensorDataset(dataset[:50000], labels[:50000])
    test_dataset = torch.utils.data.TensorDataset(dataset[50000:], labels[50000:])

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model and optimizer
    model = Classifier().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    # Train the model and save the best parameters
    best_params = None
    best_loss = 1.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = model.state_dict()
            torch.save(best_params, 'Classifier/best_model.pth')
            print("Saving best parameters...")

if __name__ == '__main__':
    main()
