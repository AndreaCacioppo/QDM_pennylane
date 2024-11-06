import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import simps
from neural_network import ROC_classifier

# Create a list to store the classifiers and their corresponding model files
classifiers = [ROC_classifier() for _ in range(10)]
model_paths = [f"ComputeMetrics/trained_models/best_model{i}.pth" for i in range(10)]

# Load the models for all 10 classifiers
for i, classifier in enumerate(classifiers):
    classifier.load_state_dict(torch.load(model_paths[i], map_location=torch.device('cpu')))
    classifier.eval()

try:
    generated_data = torch.load("Data/decoded_diffusion_dataset.pth")
    generated_labels = torch.load("Data/decoded_diffusion_labels.pth")
except:
    print("Create necessary files first")
    quit()

# shuffle data and labels
indices = torch.randperm(len(generated_data))
generated_data = generated_data[indices]
generated_labels = generated_labels[indices]

# take a small subset of the data for faster computation
generated_data = generated_data[:10000]
generated_labels = generated_labels[:10000]

axis = torch.linspace(0, 1, 5000)

# Create empty lists to store the results for all classifiers
tpr = [[] for _ in range(10)]
fpr = [[] for _ in range(10)]
auc = [0.0] * 10   

# Compute ROC curve for each classifier
for i, classifier in enumerate(classifiers):
    prob_positive = classifier(generated_data).squeeze()
    tnr, tpr_i, fnr, fpr_i = [], [], [], []
    
    for threshold in axis:
        predicted_positives = torch.where(prob_positive > threshold, torch.ones_like(prob_positive), torch.zeros_like(prob_positive))
        predicted_negatives = torch.where(prob_positive < threshold, torch.ones_like(prob_positive), torch.zeros_like(prob_positive))

        labeled_positives = torch.where(generated_labels == i, torch.ones_like(generated_labels), torch.zeros_like(generated_labels))
        labeled_negatives = torch.where(generated_labels != i, torch.ones_like(generated_labels), torch.zeros_like(generated_labels))

        true_positives = torch.sum(predicted_positives * labeled_positives)
        true_negatives = torch.sum(predicted_negatives * labeled_negatives)
        false_positives = torch.sum(predicted_positives * labeled_negatives)
        false_negatives = torch.sum(predicted_negatives * labeled_positives)

        tpr_i.append(true_positives / (true_positives + false_negatives))
        tnr.append(true_negatives / (true_negatives + false_positives))
        fpr_i.append(false_positives / (true_negatives + false_positives))
    
    tpr[i] = tpr_i
    fpr[i] = fpr_i
    auc[i] = simps(tpr_i[::-1], fpr_i[::-1])

# Print the AUC for all classifiers
for i, auc_i in enumerate(auc):
    print(f"Classifier {i} AUC: {auc_i}")

# Plot ROC curves for all classifiers
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
for i in range(10):
    plt.plot(fpr[i], tpr[i], label=f"Class {i}", color=colors[i])

plt.plot(axis, axis, label="Random classifier", linestyle='--', color='gray')
plt.xlabel("False positive rate", fontsize=14)
plt.ylabel("True positive rate", fontsize=14)
plt.title("ROC curves", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.box(True)
plt.show()
