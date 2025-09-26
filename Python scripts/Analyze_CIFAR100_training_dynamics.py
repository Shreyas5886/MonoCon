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
    is_ablation = False                        # Set to True to bypass the monotonic MLP head for the ablation study training run.

    # Evaluation parameters
    knn_k = 5                                  # Number of nearest neighbors in the k nearest neighbors (knn) algorithm

    # System parameters
    base_results_dir = './results_CIFAR100_kNN_monitoring_initial_train_dynamics' # Name of the base directory to save checkpoints and other results
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
print(f"Checkpoints will be loaded from this directory: {loading_dir}")


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
# FUNCTION FOR COMPUTING EMBEDDINGS
# =============================================================================

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


# In[ ]:


# =============================================================================
# DATASETS AND DATALOADERS
# =============================================================================
# Create train and test datasets
train_dataset = CIFAR100(root=args.data_dir, train=True, download=True, transform=eval_transform)   # Tensor-ified and normalized train dataset
test_dataset = CIFAR100(root=args.data_dir, train=False, download=True, transform=eval_transform)   # Tensor-ified and normalized test dataset

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(0.98 * num_train)
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


# ========================================================================================================
# FUNCTIONS FOR COMPUTE EFFECTIVE DIMENSIONALITY AND GENERATING AND SAVING CLUSTERMAPS
# ========================================================================================================

def get_pca_components_for_variance(data, variance_threshold=0.99):
    """
    Fits PCA to the data and returns the number of components 
    needed to explain a certain amount of variance.
    """
    # 1. Initialize and fit PCA to find all components
    pca = PCA(n_components=None, random_state=42)
    pca.fit(data)

    # 2. Calculate the cumulative sum of explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # 3. Find the number of components to reach the threshold
    # np.searchsorted finds the index where the threshold would be inserted
    # We add 1 to convert the 0-based index to a 1-based count
    n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1

    return n_components

def generate_and_save_clustermap(embeddings, title, save_path):
    """
    Computes a feature correlation matrix from embeddings, generates a clustermap,
    and saves it to the specified path.
    """
    print(f"Generating clustermap for: {title}")

    # 1. Compute the feature correlation matrix
    corr_matrix = np.corrcoef(embeddings.cpu().numpy(), rowvar=False)

    # 2. Generate the clustermap
    g = sns.clustermap(
        corr_matrix,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        figsize=(10, 10),
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )

    # 3. Set labels, title, and save the figure
    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')
    g.fig.suptitle(title, fontsize=40, y=1.04)
    g.savefig(save_path, bbox_inches='tight')

    # 4. Close the plot to free up memory
    plt.close(g.fig)

    print(f"Plot saved to {save_path}")


# In[ ]:


# ===================================================================================================================
# LOAD CHECKPOINTS, COMPUTE EFFECTIVE DIMENSIONALITY, PLOT AND SAVE CLUSTERMAPS FOR ENCODER AS WELL AS MODEL OUTPUTS
# ===================================================================================================================

model = MonoCon_ResNet(
    resnet_embedding_dim=args.resnet_embedding_dim,
    mlp_hidden_dim=args.mlp_hidden_dim,
    mlp_num_layers=args.mlp_num_layers,
    is_ablation=args.is_ablation,
).to(args.device)

model.eval()

epoch_num = list(range(1,201))

encoder_dim = np.zeros(len(epoch_num))
unnorm_mlp_dim = np.zeros(len(epoch_num))
norm_mlp_dim = np.zeros(len(epoch_num))

# --- Setup Directories for saving clustermaps ---
encoder_plot_dir = os.path.join(loading_dir, 'clustermaps_encoder')
unnorm_plot_dir = os.path.join(loading_dir, 'clustermaps_unnormalized')
norm_plot_dir = os.path.join(loading_dir, 'clustermaps_normalized')

os.makedirs(encoder_plot_dir, exist_ok=True)
os.makedirs(unnorm_plot_dir, exist_ok=True)
os.makedirs(norm_plot_dir, exist_ok=True)

for i in epoch_num:

    print(i)
    checkpoint_path = os.path.join(loading_dir, f"checkpoint_epoch_{i}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found at {checkpoint_path}.")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    #print("Model loaded successfully.")

    # Compute embeddings for encoder output, unnormalized model output, and normalized model output
    train_subset_encoder_embed, train_subset_encoder_labels = compute_embeddings(model.encoder, train_subset_loader, args.device)
    train_subset_output_embed, train_subset_output_labels = compute_embeddings(model, train_subset_loader, args.device)
    train_subset_output_embed_norm = F.normalize(train_subset_output_embed, p=2, dim=1)

    # Convert tensors to numpy arrays 
    enc_embed_np = train_subset_encoder_embed.cpu().numpy()
    out_embed_np = train_subset_output_embed.cpu().numpy()
    out_embed_norm_np = train_subset_output_embed_norm.cpu().numpy()

    # Compute effective dimensionality of the representation, defined as number of PCA components that explain 99% variance
    # PCA and clustermaps for encoder
    encoder_dim[i-1] = get_pca_components_for_variance(enc_embed_np)
    unnorm_mlp_dim[i-1] = get_pca_components_for_variance(out_embed_np)
    norm_mlp_dim[i-1] = get_pca_components_for_variance(out_embed_norm_np)

    # --- C: Generate and Save Clustermaps ---
    # Encoder Clustermap
    enc_save_path = os.path.join(encoder_plot_dir, f"clustermap_epoch_{i}.png")
    generate_and_save_clustermap(train_subset_encoder_embed, f"Encoder Clustermap - Epoch {i}", enc_save_path)

    # Unnormalized Output Clustermap
    unnorm_save_path = os.path.join(unnorm_plot_dir, f"clustermap_epoch_{i}.png")
    generate_and_save_clustermap(train_subset_output_embed, f"Unnormalized Output - Epoch {i}", unnorm_save_path)

    # Normalized Output Clustermap
    norm_save_path = os.path.join(norm_plot_dir, f"clustermap_epoch_{i}.png")
    generate_and_save_clustermap(train_subset_output_embed_norm, f"Normalized Output - Epoch {i}", norm_save_path)

    print("-" * 50) # Separator for clarity


# In[ ]:


# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
axis_label_font_size = 18
title_font_size = 20
tick_label_fontsize = 16

# --- Plot 1: Encoder Dimensionality ---
ax1.plot(epoch_num, encoder_dim, linestyle='-', label='Encoder')
ax1.set_title('Encoder',fontsize=title_font_size)
ax1.set_xlabel('Epoch', fontsize=axis_label_font_size)
ax1.set_ylabel('Effective dim', fontsize=axis_label_font_size)
ax1.tick_params(axis='both', labelsize=tick_label_fontsize)
ax1.grid(False)

# --- Plot 2: MLP Head Dimensionality ---
ax2.plot(epoch_num, unnorm_mlp_dim, linestyle='-', label='Unnormalized monotonic MLP Output')
ax2.plot(epoch_num, norm_mlp_dim, linestyle='--', label='Normalized monotonic MLP Output')
ax2.set_title('Monotonic MLP Head', fontsize=title_font_size)
ax2.set_xlabel('Epoch', fontsize=axis_label_font_size)
ax2.set_ylabel('Effective dim', fontsize=axis_label_font_size)
ax2.tick_params(axis='both', labelsize=tick_label_fontsize)
ax2.legend(fontsize = 16)
ax2.grid(False)

# Adjust layout to prevent overlap and save the figure
plt.tight_layout()

file_path = os.path.join(loading_dir, 'dimensionality_vs_epochs.png')
plt.savefig(file_path)

print(f"Plot saved to {file_path}")


# In[ ]:


print(encoder_dim.shape)


# In[ ]:




