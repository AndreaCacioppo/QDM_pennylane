import os
import sys
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE

# look outside of current directory for modules
sys.path.append(os.getcwd())

NUM_POINTS = 10000

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

# take subset of original and generated data
diffusion_points = diffusion_points[:NUM_POINTS]
diffusion_labels = diffusion_labels[:NUM_POINTS]
dataset = dataset[:NUM_POINTS]
labels = labels[:NUM_POINTS]

# merge original and generated data
all_latent_rep = torch.cat((diffusion_points, dataset))

# turn to numpy
labels = labels.detach().numpy()
latent_rep = dataset.detach().numpy()
generated_latent = diffusion_points.detach().numpy()
diffusion_labels = diffusion_labels.detach().numpy()

# perform tSNE
tsne = TSNE(n_jobs = 8)
all_latent_rep_2d = tsne.fit_transform(all_latent_rep.detach().numpy())

# Split the dataset back into the original and generated parts
generated_2d = all_latent_rep_2d[:NUM_POINTS]
latent_rep_2d = all_latent_rep_2d[NUM_POINTS:]

# save everything on file
#torch.save(generated_2d, 'Data/tsne_generated_2d.pth')
#torch.save(latent_rep_2d, 'Data/tsne_latent_rep_2d.pth')
#torch.save(labels, 'Data/tsne_latent_labels.pth')
#torch.save(diffusion_labels, 'Data/tsne_diffusion_labels.pth')

#quit()

# load everything from file
#generated_2d = torch.load('Data/tsne_generated_2d.pth')
#latent_rep_2d = torch.load('Data/tsne_latent_rep_2d.pth')
#labels = torch.load('Data/tsne_latent_labels.pth')
#diffusion_labels = torch.load('Data/tsne_diffusion_labels.pth')

# create plots for original and generated data
fig, axes = plt.subplots(1, 2, figsize=(25, 6))

# Plot original data for each digit on the first subplot
ax1 = axes[0]
for digit in range(10):
    ax1.scatter(latent_rep_2d[labels == digit, 0], latent_rep_2d[labels == digit, 1], label=f'Class {digit}', alpha=0.3)
ax1.set_title('t-SNE representation of original latent space')
ax1.set_xlabel('Component 1')
ax1.set_ylabel('Component 2')
ax1.grid(True)

# Apply Seaborn styles to the plot
sns.set(style="whitegrid")
sns.despine()

# Plot generated data for each digit on the second subplot
ax2 = axes[1]
for digit in range(10):
    ax2.scatter(generated_2d[diffusion_labels == digit, 0], generated_2d[diffusion_labels == digit, 1], label=f'Generated {digit}', alpha=0.3)
ax2.set_title('t-SNE representation of generated latent space')
ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')
ax2.legend(bbox_to_anchor=(1.3, 1.1), loc='upper right', borderaxespad=0.)
ax2.grid(True)

# Apply Seaborn styles to the plot
sns.set(style="whitegrid")
sns.despine()

plt.show()

# save plot in Results/PCA.png. If the folder does not exist, create it
if not os.path.exists('Results/'): os.makedirs('Results/')
plt.savefig('Results/TSNE.png')
