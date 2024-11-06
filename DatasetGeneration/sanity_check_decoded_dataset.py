'''visually inspect generated dataset to ensure that it is correct'''

import torch
import random
import matplotlib.pyplot as plt

# Load the dataset
dataset = torch.load('../Data/decoded_diffusion_dataset.pth')

# Get a random sample of 16 images and their labels
samples = random.sample(list(zip(dataset, torch.load('../Data/decoded_diffusion_labels.pth'))), 36)

# Create a subplot with 4 rows and 4 columns
fig, axes = plt.subplots(6, 6, figsize=(10, 6))

# Display the images and their labels
for i, (image, label) in enumerate(samples):
    row = i // 6
    col = i % 6

    ax = axes[row, col]
    ax.imshow(image.squeeze().detach().cpu().numpy(), cmap='gray')
    ax.set_title(f"Label - {label}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('sanity_check_decoded_dataset.png')
