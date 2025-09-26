#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================================
# IMPORTS
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np
import os
import shutil
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import csv


# In[ ]:


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for all hyperparameters."""
    # Data parameters
    data_dir = './data'

    # Model parameters
    resnet_embedding_dim = 512                 # ResNet34 output dimension
    mlp_hidden_dim = 1024                      # Monotonic MLP hidden dimension
    mlp_num_layers = 1                         # Monotonic MLP number of hidden layers

    # Training parameters
    batch_size = 256                           # Mini-batch size
    lr_resnet = 2e-4                           # Learning rate for the ResNet encoder
    lr_mlp = 2e-4                              # Learning rate for the monotonic MLP head
    seed = 42                                  # Random number generator seed
    temperature = 0.1
    is_ablation = True                        # Set to True to bypass the monotonic MLP head for the ablation study training run.

    # Evaluation parameters
    knn_k = 5                                  # Number of nearest neighbors in the k nearest neighbors (knn) algorithm

    # System parameters
    base_results_dir = './results_CIFAR100_kNN_monitoring'                   # Name of the base directory to save checkpoints and other results
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  # Specify whether using GPU or CPU

args = Config()                                # Create an instance of the Config class to pass arguments to functions as needed


# In[ ]:


# =============================================================================
# SETUP REPRODUCIBILITY AND DIRECTORIES
# =============================================================================
torch.manual_seed(args.seed)
np.random.seed(args.seed)

run_name = (
    f"temp_{args.temperature}_bs_{args.batch_size}_lr_encoder_{args.lr_resnet}_num_mlp_layers_{args.mlp_num_layers}"
    f"{f'_lr_head_{args.lr_mlp}' if not args.is_ablation else ''}"        # Conditionally add head learning rate
    f"{'_ablation' if args.is_ablation else ''}"       
)
loading_dir = os.path.join(args.base_results_dir, run_name)
os.makedirs(loading_dir, exist_ok=True)
print(f"Best model will be loaded from this directory: {loading_dir}")


# In[ ]:


# =============================================================================
# TRANSFORMS
# =============================================================================

# Use ImageNet's standard normalization statistics for the pre-trained ResNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.ToTensor(),                                                           # Converts image to a Pytorch tensor
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)                       # Normalizes by subtracting mean and dividing by std
])


# In[ ]:


# =============================================================================
# MODEL CLASSES
# =============================================================================

class MonotonicMLP(nn.Module):
    """Monotonic MLP with positive weights to ensure an increasing function."""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(current_dim, input_dim)

    def forward(self, x):                                   # Implement monotonicity constraints via squared weights and non-decreasing activations
        z = x   
        for layer in self.hidden_layers:
            positive_weight = layer.weight**2               # Using squared weights to ensure positivity in the hidden layers during the forward pass
            z = F.leaky_relu(F.linear(z, positive_weight, layer.bias))  # Uses the non-decreasing Leaky Relu activation function
        positive_weight_out = self.output_layer.weight**2
        z = F.linear(z, positive_weight_out, self.output_layer.bias)    # Squared weights are used for positivity in the output as well
        return z


class MonoCon_ResNet(nn.Module):
    """Metric learning model using a pre-trained ResNet and a Monotonic MLP head."""
    def __init__(self, resnet_embedding_dim, mlp_hidden_dim, mlp_num_layers, is_ablation=False):
        super().__init__()
        self.is_ablation = is_ablation                                       # Flag indicating whether to run the full model or ablation study
        self.encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)      # Initialize ResNet34 encoder with pre-trained weights
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Modify convolutional layer for low res CIFAR-100 images
        self.encoder.maxpool = nn.Identity()             # Remove max pooling layer to prevent over-compressing low res CIFAR-100 images
        self.encoder.fc = nn.Identity()                  # Remove the classification layer at the end
        if not self.is_ablation:
            self.monotonic_mlp = MonotonicMLP(resnet_embedding_dim, mlp_hidden_dim, mlp_num_layers) # Ablation study is run without the MLP head

    def forward(self, x):
        y_encoder = self.encoder(x)
        y = y_encoder if self.is_ablation else self.monotonic_mlp(y_encoder)    
        return y


# In[ ]:


# =============================================================================
# FUNCTIONS FOR COMPUTING EMBEDDINGS AND VALIDATION METRICS
# =============================================================================

def calculate_recall_at_k(query_embed, query_labels, gallery_embed, gallery_labels, k):
    """Calculates Recall@K."""
    sim_matrix = torch.matmul(query_embed, gallery_embed.T)           # Computes the cosine similarity matrix between query and gallery embeddings

    # Extract indices and labels of the top k matches (largest cosine similarity) to the query
    top_k_indices = torch.topk(sim_matrix, k=k, dim=1).indices        
    top_k_labels = gallery_labels[top_k_indices]

    # Compute the fraction of queries for which a match was found within the top k entries
    correct_recalls = (top_k_labels == query_labels.unsqueeze(1)).any(dim=1)
    recall_at_k = correct_recalls.float().mean().item()
    return recall_at_k


def compute_embeddings(model, dataloader, device):
    model.eval()
    y_list, labels_list = [], []
    pbar = tqdm(dataloader, desc="Computing embeddings", unit="batch", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels_list.append(labels.cpu())
        with torch.no_grad():
            y_batch = model(images)
        y_list.append(y_batch.cpu())
    y_embed = torch.cat(y_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return y_embed, all_labels


def evaluate_embeddings(train_embed, train_labels, test_embed, test_labels, k):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(train_embed, train_labels)
    preds = knn.predict(test_embed)
    accuracy = accuracy_score(test_labels, preds)
    print(f"k-NN Classification Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# In[ ]:


# =============================================================================
# DATASETS AND DATALOADERS
# =============================================================================
# Create train and test datasets
train_dataset = CIFAR100(root=args.data_dir, train=True, download=True, transform=eval_transform)   # Tensor-ified and normalized train dataset
test_dataset = CIFAR100(root=args.data_dir, train=False, download=True, transform=eval_transform)   # Tensor-ified and normalized test dataset

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(0.9 * num_train)
print(split)
random.seed(args.seed)
random.shuffle(indices)
subset_indices = indices[split:]

train_data_subset = Subset(train_dataset,subset_indices)
print(len(train_data_subset))

# 4. Create train and test dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
train_subset_loader =  DataLoader(train_data_subset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

print("Data loaded successfully.");


# In[ ]:


# =============================================================================
# LOAD THE BEST MODEL
# =============================================================================
model = MonoCon_ResNet(
    resnet_embedding_dim=args.resnet_embedding_dim,
    mlp_hidden_dim=args.mlp_hidden_dim,
    mlp_num_layers=args.mlp_num_layers,
    is_ablation=args.is_ablation,
).to(args.device)

best_model_path = os.path.join(loading_dir, 'best_model.pth')

if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Model file not found at {best_model_path}.")
model.load_state_dict(torch.load(best_model_path, map_location=args.device))
print("Model loaded successfully.")


# In[ ]:


# =============================================================================
# COMPUTE AND NORMALIZE TRAIN AND TEST EMBEDDINGS
# =============================================================================

# Compute embeddings
train_embed, train_labels = compute_embeddings(model, train_loader, args.device)
test_embed, test_labels = compute_embeddings(model, test_loader, args.device)
train_subset_embed, train_subset_labels = compute_embeddings(model, train_subset_loader, args.device)

######### Normalize embeddings #####################
# Uncomment tensor shape to make make sure the correct dimension is used to normalize embeddings
# print(train_embed.shape)
# print(test_embed.shape)
train_embed_norm = F.normalize(train_embed, p=2, dim=1)
test_embed_norm = F.normalize(test_embed, p=2, dim=1)
train_subset_embed_norm = F.normalize(train_subset_embed, p=2, dim=1)
print(train_subset_embed_norm.shape)
###################################################


# In[ ]:


# =============================================================================
# COMPUTE Recall@k AS A FUNCTION OF k AND SAVE NUMPY ARRAY
# =============================================================================
k_list = list(range(1,21))
recall_at_k_list = np.zeros(len(k_list))
filename  = f"recall_at_k_vs_k{'_ablation' if args.is_ablation else ''}.npy"
for j in k_list:
    recall_at_k_list[j-1] = calculate_recall_at_k(test_embed_norm, test_labels,train_embed_norm, train_labels, j)
    print(f"Recall@{j} on test embeddings is {recall_at_k_list[j-1]}")
recall_path = os.path.join(loading_dir,filename);
np.save(recall_path,recall_at_k_list)


# In[ ]:


# ================================================================================
# UNCOMMENT THIS PART TO LOAD AND PLOT Recall@k VERSUS k FOR BASELINE AND MONOCON
# ================================================================================
base_string = f"temp_{args.temperature}_bs_{args.batch_size}_lr_encoder_{args.lr_resnet}_num_mlp_layers_{args.mlp_num_layers}"
monocon_base_dir = (f"{base_string}" f"_lr_head_{args.lr_mlp}")
baseline_base_dir = (f"{base_string}" f"_ablation")
monocon_dir = os.path.join(args.base_results_dir,monocon_base_dir)
baseline_dir = os.path.join(args.base_results_dir,baseline_base_dir)
monocon_filename  = f"recall_at_k_vs_k.npy"
baseline_filename = f"recall_at_k_vs_k_ablation.npy"
monocon_path = os.path.join(monocon_dir,monocon_filename)
baseline_path = os.path.join(baseline_dir,baseline_filename)

monocon_recall_at_k = np.load(monocon_path)
baseline_recall_at_k = np.load(baseline_path)
print(len(monocon_recall_at_k))
print(len(baseline_recall_at_k))
k_list = list(range(1,len(monocon_recall_at_k)+1))
print(len(k_list))

# ================================================================================
# CREATE PLOT
# ================================================================================
plt.figure()
plt.plot(k_list, baseline_recall_at_k, marker='o', linestyle='-', label='Baseline Recall@k')
plt.plot(k_list, monocon_recall_at_k, marker='x', linestyle='--', label='MonoCon Recall@k')

# Add labels and title for clarity
plt.xlabel('k',fontsize=16)
plt.ylabel('Recall@k',fontsize=16)
#plt.title('Recall@k vs. k for Baseline and MonoCon')

# Add a legend to distinguish the lines
plt.legend(loc='lower right',fontsize=14)
plt.grid(False)
plt.savefig('recall_at_k_vs_k_plot_CIFAR-100.png')
plt.show()



# In[ ]:


# =============================================================================
# COMPUTE Recall@1 and Recall@5
# =============================================================================
final_recall_at_1 = calculate_recall_at_k(test_embed_norm, test_labels,train_embed_norm, train_labels, 1)
print(f"Final Recall@k value for k = {1} on test embeddings is {final_recall_at_1}")

final_recall_at_5 = calculate_recall_at_k(test_embed_norm, test_labels,train_embed_norm, train_labels, 5)
print(f"Final Recall@k value for k = {5} on test embeddings is {final_recall_at_5}")


# In[ ]:


# =============================================================================
# COMPUTE k-NN classification accuracy using normalized 
# =============================================================================
final_kNN_accuracy = evaluate_embeddings(train_embed_norm, train_labels, test_embed_norm, test_labels, args.knn_k)
print(f"Final k-NN classification accuracy for k = {args.knn_k} on test embeddings is {final_kNN_accuracy}")


# In[ ]:


# =============================================================================
# COMPUTE EFFECTIVE DIMENSIONALITY USING PCA ON TRAIN EMBEDDINGS
# =============================================================================
# The effective dimensionality is defined as the number of PCA components required to explain 99% variance of the train dataset

# --- Fit a full PCA on the training data to determine variance ratios ---
print("\n--- Fitting PCA on training data to analyze variance ---")
pca_full = PCA(n_components=None, random_state=42)
pca_full.fit(train_embed_norm)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_for_99_variance = np.searchsorted(cumulative_variance, 0.99) + 1
print(f"Number of components to explain 99% of TRAIN variance: {n_for_99_variance}")


# In[ ]:


# =============================================================================
# USE THIS BLOCK ONLY FOR COMPUTING METRICS FOR COMPRESSED REPRESENTATIONS
# =============================================================================

# =============================================================================
# TRUNCATE EMBEDDINGS AND EVALUATE ACROSS DIMENSIONS
# =============================================================================

# Convert initial tensors to numpy for PCA and k-NN
train_embed_np = train_embed_norm.numpy()
train_labels_np = train_labels.numpy()
test_embed_np = test_embed_norm.numpy()

# --- Loop through, create a specific PCA for each dimension, and evaluate ---
truncation_dims = [128, 64, 16]
k_for_knn = 5

for dim in truncation_dims:
    print(f"\n-- Evaluating for top {dim} PCA components --")

    # 1. Define and fit a new PCA model for the current dimension
    pca_reconstruction = PCA(n_components=dim, random_state=42)
    pca_reconstruction.fit(train_embed_np) # Fit on NumPy training data

    # 2. Calculate Reconstruction Error 
    # Reconstruct the test data by transforming and inverse_transforming
    reconstructed_test_np = pca_reconstruction.inverse_transform(pca_reconstruction.transform(test_embed_np))

    # Convert back to PyTorch Tensors for RMSE calculation
    original_test_torch = torch.from_numpy(test_embed_np)
    reconstructed_test_torch = torch.from_numpy(reconstructed_test_np)

    # Implement the RMSE formula
    reconstruction_rmse = torch.sqrt(torch.mean(torch.square(original_test_torch - reconstructed_test_torch)))
    print(f"PCA Reconstruction RMSE: {reconstruction_rmse.item():.8f}")

    # 3. Get the truncated embeddings for downstream tasks
    train_embed_truncated = pca_reconstruction.transform(train_embed_np)
    test_embed_truncated = pca_reconstruction.transform(test_embed_np)

    # 4. Evaluate k-NN Classification Accuracy (uses NumPy arrays)
    evaluate_embeddings(train_embed_truncated, train_labels_np, test_embed_truncated, test_labels.numpy(), k=k_for_knn)

    # 5. Evaluate Recall@k (uses PyTorch Tensors)
    # Convert the truncated embeddings back to Tensors
    train_embed_truncated_torch = torch.from_numpy(train_embed_truncated)
    test_embed_truncated_torch = torch.from_numpy(test_embed_truncated)

    # Use TEST set as QUERY and TRAIN set as GALLERY
    recall_at_1 = calculate_recall_at_k(test_embed_truncated_torch, test_labels, train_embed_truncated_torch, train_labels, k=1)
    recall_at_5 = calculate_recall_at_k(test_embed_truncated_torch, test_labels, train_embed_truncated_torch, train_labels, k=5)

    print(f"Recall@1: {recall_at_1 * 100:.2f}%")
    print(f"Recall@5: {recall_at_5 * 100:.2f}%")


# In[ ]:


# ===============================================================================
# QUANTIFY ROBUSTNESS USING PCA-BASED RMS RECONSTRUCTION ERROR ON TEST EMBEDDINGS
# ===============================================================================

# 1. Define a new PCA model with number of components equal to effective dimensionality defined in the previous cell
pca_reconstruction = PCA(n_components=n_for_99_variance, random_state=42)

# 2. Fit it on the training data
pca_reconstruction.fit(train_embed_norm)

# 3. Reconstruct the test data by transforming and inverse_transforming
test_reconstructed = pca_reconstruction.inverse_transform(pca_reconstruction.transform(test_embed_norm))

# 4. Calculate the Root Mean Squared Error between original and reconstructed test data
reconstruction_rmse = torch.sqrt(torch.mean(torch.square(test_embed_norm - test_reconstructed)))
print(f"PCA-based Root Mean Squared Reconstruction Error on test embeddings is: {reconstruction_rmse:.8f}")


# In[ ]:


# =============================================================================
# COMPUTE FEATURE CORRELATION MATRIX, AND VISUALIZE AND STORE ITS CLUSTERMAP
# =============================================================================

print(f""" Generating feature covariance matrix computed using normalized output features for a fixed random subset of training data""")

feat_corr_matrix = np.corrcoef(train_subset_embed_norm.to('cpu').numpy(), rowvar=False)

# Generate Correlation Clustermap

print(f""" Generating clustermap of feature covariance matrix""")

path_corr = os.path.join(loading_dir, f"best_model_norm_output_feature_covariance_matrix.png")

g = sns.clustermap(
    feat_corr_matrix,
    cmap='RdBu_r',  # Red-Blue diverging colormap
    vmin=-1,
    vmax=1,
    figsize=(10, 10),
    cbar_pos=(0.02, 0.8, 0.03, 0.15) # Position colorbar
)

g.ax_heatmap.set_xlabel('')
g.ax_heatmap.set_ylabel('')
g.fig.suptitle('MonoCon', fontsize=50, y=1.06)
g.savefig(path_corr)

print(f"Plot saved to {path_corr}")


# In[ ]:


# ===================================================================================
# COMPUTE ROOT MEAN SQUARE OFF-DIAGONAL CORRELATION IN THE FEATURE CORRELATION MATRIX
# ===================================================================================
embed_dim = feat_corr_matrix.shape[0]
mask = 1.0 - np.eye(embed_dim)
rms_off_diagonal_feat_corr = np.sqrt(np.sum(np.square(feat_corr_matrix*mask))/(embed_dim*(embed_dim-1)))
print(f"RMS Off diagonal correlation in the feature correlation matrix is: {rms_off_diagonal_feat_corr:.4f}")


# In[ ]:


# ===================================================================================
# SAVE PERFORMANCE METRICS TO A JSON FILE
# ===================================================================================

import json

performance_metrics = {
    "kNN accuracy for k=5 (%)": float(final_kNN_accuracy*100),
    "Recall@1 (%)": float(final_recall_at_1*100),
    "Recall@5 (%)": float(final_recall_at_5*100),
    "number of PCA components for 99% variance": int(n_for_99_variance),
    "PCA-based RMS reconstruction error": float(reconstruction_rmse),
    "RMS off-diagonal feature correlation": float(rms_off_diagonal_feat_corr)
}

json_file_path = os.path.join(loading_dir, 'best_model_performance_metrics.json')
with open(json_file_path, "w") as f:
    json.dump(performance_metrics, f, indent=4)

print(f"Performance metrics saved to {json_file_path}")

