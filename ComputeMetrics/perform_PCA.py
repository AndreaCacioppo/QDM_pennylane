import os
import sys
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# look outside of current directory for modules
sys.path.append(os.getcwd())

# TODO: 
# - add averages
# - sanity check results

# load latent space and corresponding labels
try:
    dataset = torch.load('Data/latent_mnist.pth')
    labels = torch.load('Data/latent_labels.pth')
    diffusion_points = torch.load('Data/latent_diffusion_dataset.pth')
    diffusion_labels = torch.load('Data/latent_diffusion_labels.pth')
except FileNotFoundError: 
    print("Create necessary files first")
    quit()

# take a subsample of diffusion points and corresponding labels, first shuffle
indices = torch.randperm(len(diffusion_points))
diffusion_points = diffusion_points[indices]
diffusion_labels = diffusion_labels[indices]
dataset = dataset[indices]
labels = labels[indices]

len_generated = diffusion_points.shape[0]

# merge original and generated data
all_latent_rep = torch.cat((dataset, dataset))#torch.cat((diffusion_points.detach(), dataset))

# turn to numpy
labels = labels.detach().numpy()
latent_rep = dataset.detach().numpy()
generated_latent = diffusion_points.detach().numpy()

# Perform PCA on the joint dataset
pca = PCA(n_components=2)
all_latent_rep_2d = pca.fit_transform(all_latent_rep.detach().numpy())

# Split the dataset back into the original and generated parts
generated_2d = all_latent_rep_2d[:len_generated]
latent_rep_2d = all_latent_rep_2d[len_generated:]

# take subset of original data and generated data
latent_rep_2d = latent_rep_2d[:10000]
generated_2d = generated_2d[:10000]
labels = labels[:10000]
diffusion_labels = diffusion_labels[:10000].detach().numpy()

# create plots for original and generated data
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Plot original data for each digit on the first subplot
ax1 = axes[0]
for digit in range(10):
    ax1.scatter(latent_rep_2d[labels == digit, 0], latent_rep_2d[labels == digit, 1], label=f'Class {digit}', alpha=0.3)
ax1.set_title('PCA representation of original latent space')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_xlim([-0.5, 0.5])
ax1.set_ylim([-0.5, 0.5])
ax1.legend()
ax1.grid(True)

# Apply Seaborn styles to the plot
sns.set(style="whitegrid")
sns.despine()

# Plot generated data for each digit on the second subplot
ax2 = axes[1]
for digit in range(10):
    ax2.scatter(generated_2d[diffusion_labels == digit, 0], generated_2d[diffusion_labels == digit, 1], label=f'Generated {digit}', alpha=0.3)
ax2.set_title('PCA representation of generated latent space')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
ax1.set_xlim([-0.5, 0.5])
ax1.set_ylim([-0.5, 0.5])
ax2.legend()
ax2.grid(True)

# Apply Seaborn styles to the plot
sns.set(style="whitegrid")
sns.despine()

plt.show()

# save plot in Results/PCA.png. If the folder does not exist, create it
if not os.path.exists('Results/'): os.makedirs('Results/')
plt.savefig('Results/PCA.png')
