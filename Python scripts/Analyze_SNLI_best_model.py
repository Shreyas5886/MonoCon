#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================================
# IMPORTS
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import shutil
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import random


# In[ ]:


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for all hyperparameters."""
    # Model parameters
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    lm_embedding_dim = 384      # MiniLM's output dimension
    mlp_hidden_dim = 768        # Monotonic MLP hidden dimension
    mlp_num_layers = 1          # Monotonic MLP number of hidden layers
    max_seq_length = 128        # Max sequence length for tokenizer

    # Training parameters
    epochs = 40                 # Maximum number of epochs used for training
    patience = 10               # Number of epochs after which early stopping is triggered
    batch_size = 128            # Mini-batch size
    lr_lm = 2e-7                # Learning rate for fine-tuning the language model encoder
    lr_mlp = 2e-4               # Learning rate for the monotonic MLP head 
    temperature = 0.05          # Temperature parameter for SupCon loss. Controls the importance of hard negatives in the learning task
    seed = 42                   # random number generator seed
    is_ablation = False         # Set to True to bypass the monotonic MLP head for the ablation study training run.

    # System parameters
    base_results_dir = './results_SNLI' # Only the base path is needed here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Config()                  # Create an instance of the Config class to pass arguments to functions as needed


# In[ ]:


# =============================================================================
# SETUP REPRODUCIBILITY AND DIRECTORIES
# =============================================================================
torch.manual_seed(args.seed)
np.random.seed(args.seed)

run_name = (
    f"temp_{args.temperature}_bs_{args.batch_size}_lr_encoder_{args.lr_lm}_epochs_{args.epochs}_patience_{args.patience}"
    f"{f'_lr_head_{args.lr_mlp}' if not args.is_ablation else ''}"  # Conditionally add head learning rate
    f"{'_ablation' if args.is_ablation else ''}"
)
loading_dir = os.path.join(args.base_results_dir, run_name)
os.makedirs(loading_dir, exist_ok=True)
print(f"Best model will be loaded from this directory: {loading_dir}")


# In[ ]:


# =============================================================================
# MODEL CLASSES
# =============================================================================

class MonotonicMLP(nn.Module):
    """Monotonic MLP implemented using positive weights and nondecreasing activation functions."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):  
        super().__init__()
        self.num_layers = num_layers
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(current_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # Implement monotonicity constraints via squared weights and non-decreasing activations
        z = x
        for layer in self.hidden_layers:
            positive_weight = layer.weight**2    # Using squared weights to ensure positivity in the hidden layers during the forward pass
            z = F.leaky_relu(F.linear(z, positive_weight, layer.bias))  # Uses the non-decreasing Leaky Relu activation function
        positive_weight_out = self.output_layer.weight**2
        z = F.linear(z, positive_weight_out, self.output_layer.bias)    # Squared weights are used for positivity in the output as well
        return z

class MonoCon_LM(nn.Module):
    """Metric learning model using a pre-trained LM and a Monotonic MLP head."""
    def __init__(self, model_name: str, lm_embedding_dim: int, mlp_hidden_dim: int, mlp_num_layers: int, is_ablation: bool = False):
        super().__init__()
        self.is_ablation = is_ablation      # Flag indicating whether to run the full model or ablation study
        self.encoder = AutoModel.from_pretrained(model_name)     # Sentence transformer encoder (MiniLM) specified in the Config class
        if not self.is_ablation:
            self.monotonic_mlp = MonotonicMLP(lm_embedding_dim, mlp_hidden_dim, mlp_num_layers)  # Monotonic MLP is created only for the full model

    def _mean_pooling(self, model_output, attention_mask):    # Generates a single average vector from multiple contextual embedding vectors 
        token_embeddings = model_output[0]                    # Extracts token embeddings from the model output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()   # Accounts for variable sentence lengths by using padded dimensions that don't contribute to averaging
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, **kwargs):
        y_encoder_tokens = self.encoder(**kwargs)     # The MiniLM encoder generates sequences of tokens with contextual embeddings for each token
        y_encoder = self._mean_pooling(y_encoder_tokens, kwargs['attention_mask'])    # Averages the contextual embeddings to yield a single sentence embedding vector 
        y = y_encoder if self.is_ablation else self.monotonic_mlp(y_encoder)          # MLP is only used to get embeddings for the full model
        return y

class SNLIDataset(Dataset):
    """Custom PyTorch dataset that contains only those sentence pairs from the full SNLI dataset that have an entailment relationship. 
       These constitute positive pairs for the purposes of supervised contrastive learning"""
    def __init__(self, data):
        self.samples = []
        for example in data:
            if example['label'] == 0: # The label 0 indicates entailment
                self.samples.append((example['premise'], example['hypothesis']))

    def __len__(self):      # Returns the total number of entailment pairs, i.e. the total dataset size
        return len(self.samples)

    def __getitem__(self, idx):    # Returns a sentence pair based on its index
        return self.samples[idx]


# In[ ]:


# =============================================================================
# FUNCTIONS TO LOAD DATA AND COMPUTE EMBEDDINGS
# =============================================================================
def create_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        premises = [item[0] for item in batch]
        hypotheses = [item[1] for item in batch]
        sentences = premises + hypotheses
        tokenized = tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        labels = torch.arange(len(premises)).repeat(2)
        return tokenized, labels
    return collate_fn


def compute_embeddings_for_stsb(model, tokenizer, stsb_data, device, args):
    model.eval()
    sents1 = [ex['sentence1'] for ex in stsb_data]
    sents2 = [ex['sentence2'] for ex in stsb_data]

    @torch.no_grad()
    def get_embeds(sentences):
        all_embeds = []
        pbar_embed = tqdm(range(0, len(sentences), args.batch_size), desc="Computing embeddings", leave=False)
        for start_idx in pbar_embed:
            batch_sents = sentences[start_idx : start_idx + args.batch_size]
            tokenized = tokenizer(batch_sents, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt')
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            y_batch = model(**tokenized)
            all_embeds.append(y_batch.cpu())
        return torch.cat(all_embeds, dim=0)

    embeds1 = get_embeds(sents1)
    embeds2 = get_embeds(sents2)
    return embeds1, embeds2


def evaluate_stsb(model, tokenizer, stsb_data, args, silent=False):
    if not silent:
        print(f"\n--- Evaluating on STSb Benchmark ---")

    embeddings1, embeddings2 = compute_embeddings_for_stsb(model, tokenizer, stsb_data, args.device, args)

    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    cosine_scores = torch.sum(embeddings1 * embeddings2, dim=1)
    labels = torch.tensor([ex['score'] for ex in stsb_data])

    spearman_corr, _ = spearmanr(cosine_scores.numpy(), labels.numpy())

    if not silent:
        print(f"Spearman Correlation: {spearman_corr:.4f}")
    return spearman_corr


@torch.no_grad(   )
def generate_embeddings(sentences, model, tokenizer, args):
    """Computes embeddings for a given list of sentences."""
    model.eval()
    all_embeds = []
    # Create a progress bar for embedding generation
    pbar_embed = tqdm(
        range(0, len(sentences), args.batch_size),
        desc=f"Computing embeddings for {len(sentences)} sentences",
        leave=False
    )
    for start_idx in pbar_embed:
        batch_sents = sentences[start_idx : start_idx + args.batch_size]
        tokenized = tokenizer(
            batch_sents,
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors='pt'
        )
        tokenized = {k: v.to(args.device) for k, v in tokenized.items()}
        y_batch = model(**tokenized)
        all_embeds.append(y_batch.cpu())
    return torch.cat(all_embeds, dim=0)


# In[ ]:


# =============================================================================
# DATASETS AND DATALOADERS
# =============================================================================

print("--- Loading tokenizer and datasets ---")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

snli_dataset_raw = load_dataset('snli', split='train').filter(lambda ex: ex['label'] in [0, 1, 2])
train_dataset = SNLIDataset(snli_dataset_raw)
collate_fn = create_collate_fn(tokenizer, args.max_seq_length)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=0)

stsb_val_data = list(load_dataset('sentence-transformers/stsb', split='validation'))
stsb_test_data = list(load_dataset('sentence-transformers/stsb', split='test'))

print("Data loaded successfully.");


# In[ ]:


# =============================================================================
# LOAD THE BEST MODEL
# =============================================================================

model = MonoCon_LM(
    model_name=args.model_name,
    lm_embedding_dim=args.lm_embedding_dim,
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
# COMPUTE STS-B SCORE (SPEARMAN CORRELATION COEFFICIENT)
# =============================================================================

stsb_spearman = 100*evaluate_stsb(model, tokenizer, stsb_test_data, args)
print(f"STS-B Spearman Coefficient Score is:{stsb_spearman}")


# In[ ]:


# =============================================================================
# GENERATE AND NORMALIZE EMBEDDINGS FOR TRAIN AND TEST SETS
# =============================================================================

model.eval()

# --- Generate Test Embeddings (STSb) ---
print("Generating embeddings for the TEST dataset (STSb)...")
sents1_test = [ex['sentence1'] for ex in stsb_test_data]
sents2_test = [ex['sentence2'] for ex in stsb_test_data]
unique_test_sentences = list(set(sents1_test + sents2_test))
test_embeddings_tensor = generate_embeddings(unique_test_sentences, model, tokenizer, args)
# Normalize embeddings for PCA, as it's sensitive to vector magnitude
test_embed = F.normalize(test_embeddings_tensor, p=2, dim=1).numpy()
print(f"Generated {test_embed.shape[0]} unique test embeddings of dimension {test_embed.shape[1]}.")

# --- Generate Train Embeddings (SNLI) ---
print("Generating embeddings for the TRAIN dataset (SNLI)...")
train_sentences = list(set([s for p, h in train_dataset.samples for s in (p, h)]))
train_embeddings_tensor = generate_embeddings(train_sentences, model, tokenizer, args)
print(train_embeddings_tensor.shape)

# --- Preparing a fixed random subset of Train Embeddings (SNLI) for feature correlation matrix analysis ---
print("Preparing fixed random subset of Train Embeddings (SNLI) for feature correlation matrix analysis...") 
num_train = len(train_embeddings_tensor[:,1])
indices = list(range(num_train))
split = int(0.9 * num_train)
print(split)
random.seed(args.seed)
random.shuffle(indices)
subset_indices = indices[split:]

train_subset_embeddings_tensor = train_embeddings_tensor[subset_indices,:]

# Normalize embeddings
train_embed_norm = F.normalize(train_embeddings_tensor, p=2, dim=1).numpy()
train_subset_embed_norm = F.normalize(train_subset_embeddings_tensor, p=2, dim=1).numpy()
print(f"Generated {train_embed_norm.shape[0]} unique train embeddings of dimension {train_embed_norm.shape[1]}.")


# In[ ]:


# =============================================================================
# COMPUTE EFFECTIVE DIMENSIONALITY USING PCA ON TRAIN EMBEDDINGS
# =============================================================================
# The effective dimensionality is defined as the number of PCA components required to explain 99% variance of the train dataset

# --- Fit a full PCA on the training data to find the optimal number of components ---
print("\n--- Fitting PCA on training data to analyze variance ---")
pca_full = PCA(n_components=None, random_state=42)
pca_full.fit(train_embed_norm)

# Find the number of components that explain 99% of the variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_for_99_variance = np.searchsorted(cumulative_variance, 0.99) + 1
print(f"Number of components to explain 99% of TRAIN variance: {n_for_99_variance}")


# In[ ]:


# ===============================================================================
# QUANTIFY ROBUSTNESS USING PCA-BASED RMS RECONSTRUCTION ERROR ON TEST EMBEDDINGS
# ===============================================================================

print(f"\n--- Calculating Test Set Reconstruction Error using top {n_for_99_variance} components from train subspace ---")

# 1. Define a new PCA model with number of components equal to effective dimensionality defined in the previous cell
pca_reconstruction = PCA(n_components=n_for_99_variance, random_state=42)

# 2. Fit this PCA model ONLY on the training data to define the subspace
pca_reconstruction.fit(train_embed_norm)

# 3. Reconstruct the test data by projecting it onto the train subspace and then inverting
test_projected = pca_reconstruction.transform(test_embed)
test_reconstructed = pca_reconstruction.inverse_transform(test_projected)

# 4. Calculate the Root Mean Squared Error between original and reconstructed test data
reconstruction_rmse = np.sqrt(np.mean(np.square(test_embed - test_reconstructed)))
print(f"Test Set Reconstruction RMSE from Train Subspace: {reconstruction_rmse:.8f}")


# In[ ]:


# =============================================================================
# COMPUTE FEATURE CORRELATION MATRIX, AND VISUALIZE AND STORE ITS CLUSTERMAP
# =============================================================================
import seaborn as sns

print(f""" Generating feature covariance matrix computed using normalized output features for a fixed random subset of training data""")

feat_corr_matrix = np.corrcoef(train_subset_embed_norm, rowvar=False)

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
    "STSb_Spearman": float(stsb_spearman),
    "number of PCA components for 99% variance": int(n_for_99_variance),
    "PCA-based RMS reconstruction error": float(reconstruction_rmse),
    "RMS off-diagonal feature correlation": float(rms_off_diagonal_feat_corr)
}

json_file_path = os.path.join(loading_dir, 'best_model_performance_metrics.json')
with open(json_file_path, "w") as f:
    json.dump(performance_metrics, f, indent=4)

print(f"Performance metrics saved to {json_file_path}")


# In[ ]:




