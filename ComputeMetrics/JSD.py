'''compute KL divergence bewteen two datasets and corresponding
latent datasets, loaded from Data to be used as a quality metric'''

import torch
from sklearn.decomposition import PCA

NUM_COMPONENTS = 4 # number of components for PCA
NUM_BINS = 20

# image space distributions (original and generated)
image_origin = torch.load("Data/mnist.pth").flatten(1, -1)
image_labels_origin = torch.load("Data/labels.pth")
image_target = torch.load("Data/decoded_diffusion_dataset.pth").flatten(1, -1)
image_labels_target = torch.load("Data/decoded_diffusion_labels.pth")

# reduce dimensionality of image dataset using PCA
pca = PCA(n_components = NUM_COMPONENTS)
origin_and_target = torch.cat((image_origin, image_target)).detach().numpy()
pca.fit(origin_and_target)

# compress image data
compressed_origin = torch.tensor(pca.transform(image_origin.detach().numpy()))
compressed_target = torch.tensor(pca.transform(image_target.detach().numpy()))

# split digits
# Define an empty list to store KL divergences for image and latent spaces
JSD_image = []
epsilon = 1e-10

# Iterate over digits from 0 to 9
for digit in range(10):

    # Filter images and latents for the current digit
    image_origin_digit = compressed_origin[image_labels_origin == digit]
    image_target_digit = compressed_target[image_labels_target == digit]

    # Compute histograms
    hist_image_origin_digit = torch.histogramdd(image_origin_digit, bins=(NUM_BINS)*NUM_COMPONENTS)[0]
    hist_image_target_digit = torch.histogramdd(image_target_digit, bins=(NUM_BINS)*NUM_COMPONENTS)[0]

    # Normalize histograms
    hist_image_origin_digit = hist_image_origin_digit / (torch.sum(hist_image_origin_digit) + epsilon)
    hist_image_target_digit = hist_image_target_digit / (torch.sum(hist_image_target_digit) + epsilon)

    # compute mixture
    mixture_image = (hist_image_origin_digit + hist_image_target_digit) / 2

    # Compute JSD
    JSD_digit = (torch.sum(hist_image_origin_digit * torch.log(hist_image_origin_digit / mixture_image)) + torch.sum(hist_image_target_digit * torch.log(hist_image_target_digit / mixture_image))) / 2

    # Append the results to the respective lists
    JSD_image.append(JSD_digit.detach().numpy())

    print("=================================================")
    print("KL divergence for digit", digit, "in image space: ", JSD_image[digit])
    print("=================================================")
