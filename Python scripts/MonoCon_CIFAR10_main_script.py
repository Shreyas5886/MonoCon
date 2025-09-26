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
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np
import os
import shutil
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
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
    epochs = 200                               # Maximum number of epochs used for training
    patience = 20                              # Number of epochs after which early stopping is triggered
    batch_size = 256                           # Mini-batch size
    warmup_epochs = 10                         # Number of epochs to warm up the head. Set to 0 to disable.
    lr_warmup = 1e-4                           # Learning rate specifically for the warm-up phase.
    lr_resnet = 5e-4                           # Learning rate for the ResNet encoder
    lr_mlp = 5e-4                              # Learning rate for the monotonic MLP head
    weight_decay = 1e-4                        # Weight decay (L2) regularization
    temperature = 0.1                          # Temperature parameter for SupCon loss. Controls the importance of hard negatives in the learning task
    grad_clip_norm = 1.0                       # Max norm of gradients. Leads to more stable training
    seed = 42                                  # Random number generator seed
    is_ablation = True                        # Set to True to bypass the monotonic MLP head for the ablation study training run.

    # Evaluation parameters
    knn_k = 5                                  # Number of nearest neighbors in the k nearest neighbors (knn) algorithm
    recall_k = 1                               # Number of top matches for the recall@k metric
    validation_frequency = 5                   ## frequency (in number of epochs) with which validation metrics are calculated

    # System parameters
    base_results_dir = './results_CIFAR10_kNN_monitoring'                   # Name of the base directory to save checkpoints and other results
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  # Specify whether using GPU or CPU

args = Config()                                # Create an instance of the Config class to pass arguments to functions as needed


# In[ ]:


# =============================================================================
# SETUP REPRODUCIBILITY AND DIRECTORIES
# =============================================================================
print(f"Using device: {args.device}")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True               # Use only deterministic algorithms in the cuDNN backend
torch.backends.cudnn.benchmark = False                  # Prevents cuDNN from potentially picking a fast but non-deterministic algorithm.

run_name = (
    f"temp_{args.temperature}_bs_{args.batch_size}_lr_encoder_{args.lr_resnet}_num_mlp_layers_{args.mlp_num_layers}"
    f"{f'_lr_head_{args.lr_mlp}' if not args.is_ablation else ''}"        # Conditionally add head learning rate
    f"{'_ablation' if args.is_ablation else ''}"       
)
output_dir = os.path.join(args.base_results_dir, run_name)
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory for this run: {output_dir}")


# In[ ]:


# =============================================================================
# TRANSFORMS
# =============================================================================

# Use ImageNet's standard normalization statistics for the pre-trained ResNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Note: Order of composition of transforms is important. Operations must be performed on correct data types. 
# For eg. .ToTensor() must precede .Normalize() as normalization must be performed on pytorch tensors, not images.
train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(),                                                 # Applies one out of a wide range of transformations randomly
    transforms.ToTensor(),                                                           # Converts image to a Pytorch tensor
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),  # Erases pixels in a random rectangle with probability p
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)                       # Normalizes by subtracting mean and dividing by std
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


# In[ ]:


# =============================================================================
# MODEL AND DATA CLASSES
# =============================================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss function"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device                                     # Identify the device on which the features tensor is stored
        batch_size = features.shape[0]                               # Extract the batch_size. Useful for later operations
        labels = labels.contiguous().view(-1,1)                      # Assign a continuous memory block to sotre labels. Reshape the labels tensor
        mask = torch.eq(labels,labels.T).float().to(device)          # Creates a square matrix mask via broadcasting, to identify positive pairs
        anchor_dot_contrast = torch.div(torch.matmul(features,features.T),self.temperature)   # Matrix of scaled feature dot products
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)          # Compute maximum value of the scaled dot product in each row
        # The next line subtracts a constant from each row, which leaves the loss unchanged but prevents numerical instability
        logits = anchor_dot_contrast - logits_max.detach()
        # The following line creates a mask with diagonal elements 0 and the rest 1. 
        # Implementation using torch.scatter instead of the simple torch.eye leads to smoother training and better learned representations
        logits_mask = torch.scatter(                                 
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0  
        )                                           
        mask = mask*logits_mask                     # Removes diagonal elements, so that positive pairs of samples with themselves are not counted

        # ====== Main SupCon loss calculation =============
        exp_logits = torch.exp(logits)*logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask*log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        # =================================================
        return loss


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
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Modify convolutional layer for low res CIFAR-10 images
        self.encoder.maxpool = nn.Identity()             # Remove max pooling layer to prevent over-compressing low res CIFAR-10 images
        self.encoder.fc = nn.Identity()                  # Remove the classification layer at the end
        if not self.is_ablation:
            self.monotonic_mlp = MonotonicMLP(resnet_embedding_dim, mlp_hidden_dim, mlp_num_layers) # Ablation study is run without the MLP head

    def forward(self, x):
        y_encoder = self.encoder(x)
        y = y_encoder if self.is_ablation else self.monotonic_mlp(y_encoder)    
        return y


# In[ ]:


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_recall_at_k(query_embed, query_labels, gallery_embed, gallery_labels, k):
    """Calculates Recall@K."""
    query_embed = F.normalize(query_embed, p=2, dim=1)                           
    gallery_embed = F.normalize(gallery_embed, p=2, dim=1)
    sim_matrix = torch.matmul(query_embed, gallery_embed.T)           # Computes the cosine similarity matrix between query and gallery embeddings

    # Extract indices and labels of the top k matches (largest cosine similarity) to the query
    top_k_indices = torch.topk(sim_matrix, k=k, dim=1).indices        
    top_k_labels = gallery_labels[top_k_indices]

    # Compute the fraction of queries for which a match was found within the top k entries
    correct_recalls = (top_k_labels == query_labels.unsqueeze(1)).any(dim=1)
    recall_at_k = correct_recalls.float().mean().item()
    return recall_at_k


def run_validation_metrics(model, val_loader, gallery_loader, criterion, args):
    """Runs a full validation, returning loss, k-NN accuracy, and Recall@K."""
    model.eval()

    #  Compute gallery and query embeddings for quantifying recall@k score
    gallery_embed, gallery_labels = compute_embeddings(model, gallery_loader, args.device)
    query_embed, query_labels = compute_embeddings(model, val_loader, args.device)

    # Compute the kNN classification accuracy
    knn_accuracy = evaluate_embeddings(
        F.normalize(gallery_embed, p=2, dim=1).numpy(), gallery_labels.numpy(),
        F.normalize(query_embed, p=2, dim=1).numpy(), query_labels.numpy(),
        args.knn_k, "Validation k-NN"
    )

    # Compute the recall@k score
    recall_at_k = calculate_recall_at_k(query_embed, query_labels, gallery_embed, gallery_labels, k=args.recall_k)

    # Compute the average validation loss
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            y_batch_norm = F.normalize(model(images), p=2, dim=1)
            loss = criterion(y_batch_norm, labels)
            total_loss += loss.item()
    avg_val_loss = total_loss / len(val_loader)

    return {
        'val_loss': avg_val_loss,
        'knn_acc': knn_accuracy,
        f'recall@{args.recall_k}': recall_at_k
    }


def compute_embeddings(model, dataloader, device, baseline_resnet_only=False):
    model.eval()
    y_list, labels_list = [], []
    pbar = tqdm(dataloader, desc="Computing embeddings", unit="batch", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels_list.append(labels.cpu())
        with torch.no_grad():
            if baseline_resnet_only:
                y_batch = model.encoder(images)
            else:
                y_batch = model(images)
        y_list.append(y_batch.cpu())
    y_embed = torch.cat(y_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return y_embed, all_labels


def evaluate_embeddings(train_embed, train_labels, test_embed, test_labels, k, name):
    print(f"\n--- Evaluating {name} embeddings (k={k}) ---")
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(train_embed, train_labels)
    preds = knn.predict(test_embed)
    accuracy = accuracy_score(test_labels, preds)
    print(f"k-NN Classification Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def visualize_embeddings(embeddings, labels, output_dir, seed, name_prefix):
    print(f"\n--- Visualizing {name_prefix} embeddings ---")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=seed, n_jobs=-1)
    embeddings_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(14, 12))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='hsv', s=12, alpha=0.7)
    plt.title(f"t-SNE of {name_prefix} Embeddings on CIFAR-10", fontsize=16)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    filename = f"{name_prefix}_cifar10_tsne.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    plt.close()
    print(f"Saved plot to: {os.path.join(output_dir, filename)}")


# In[ ]:


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def run_epoch(model, dataloader, criterion, optimizer, args, epoch_desc, params_to_clip):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=epoch_desc, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        y_batch = model(images)
        y_batch_norm = F.normalize(y_batch, p=2, dim=1)
        loss = criterion(y_batch_norm, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=args.grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, gallery_loader, criterion, optimizer, scheduler, args, output_dir):
    """Main training loop with robust checkpointing, CSV logging, and periodic validation."""
    start_epoch = 0
    best_knn_acc = 0.0
    epochs_no_improve = 0

    best_model_path = os.path.join(output_dir, 'best_model.pth')
    latest_checkpoint_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    log_file_path = os.path.join(output_dir, 'training_log.csv')
    final_best_model_path = None

    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_knn_acc = checkpoint.get('best_knn_acc', 0.0) 
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"   Resumed from epoch {start_epoch}. Best k-NN accuracy: {best_knn_acc*100:.2f}%")
        print(f"   Patience counter (epochs with no improvement): {epochs_no_improve}")
    else:
        print("Starting training from scratch.")
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['epoch', 'train_loss', 'val_loss', 'knn_acc', f'recall@{args.recall_k}', 'lr_encoder', 'lr_head']
            if args.is_ablation:
                header = header[:-1]
            writer.writerow(header)

    for epoch in range(start_epoch, args.epochs):
        epoch_desc = f"Epoch {epoch+1}/{args.epochs}"
        avg_train_loss = run_epoch(model, train_loader, criterion, optimizer, args, epoch_desc, model.parameters())
        scheduler.step()

        if (epoch + 1) % args.validation_frequency == 0 or (epoch + 1) == args.epochs:
            val_metrics = run_validation_metrics(model, val_loader, gallery_loader, criterion, args)

            if not args.is_ablation:
                lr_log_str = f"LRs (ResNet/Head): {optimizer.param_groups[0]['lr']:.1e}/{optimizer.param_groups[1]['lr']:.1e}"
            else:
                lr_log_str = f"LR (ResNet): {optimizer.param_groups[0]['lr']:.1e}"

            print(f"Epoch {epoch+1}/{args.epochs} -> "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val k-NN: {val_metrics['knn_acc']*100:.2f}% | "
                  f"Val Recall@{args.recall_k}: {val_metrics[f'recall@{args.recall_k}']*100:.2f}% | "
                  f"{lr_log_str}")

            if val_metrics['knn_acc'] > best_knn_acc:
                best_knn_acc = val_metrics['knn_acc']
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                final_best_model_path = best_model_path
                print(f"New best model saved with Val k-NN Acc: {best_knn_acc*100:.2f}%")
            else:
                epochs_no_improve += args.validation_frequency

            current_epoch_checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'best_knn_acc': best_knn_acc, 'epochs_no_improve': epochs_no_improve,
            }, current_epoch_checkpoint_path)
            shutil.copyfile(current_epoch_checkpoint_path, latest_checkpoint_path)

            with open(log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                lr_encoder = optimizer.param_groups[0]['lr']
                row_data = {
                    'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': val_metrics['val_loss'],
                    'knn_acc': val_metrics['knn_acc'], f'recall@{args.recall_k}': val_metrics[f'recall@{args.recall_k}'],
                    'lr_encoder': lr_encoder
                }
                if not args.is_ablation:
                    row_data['lr_head'] = optimizer.param_groups[1]['lr']
                writer.writerow(list(row_data.values()))

            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement on k-NN accuracy.")
                break
        else:
            print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}")

    print("--- Training Finished ---")
    return final_best_model_path


# In[ ]:


# =============================================================================
# DATASETS AND DATALOADERS
# =============================================================================

# 1. Create two base datasets from the same training data but with different transforms
train_dataset_aug = CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
train_dataset_no_aug = CIFAR10(root=args.data_dir, train=True, download=True, transform=test_transform)

# 2. Create a single set of indices to split the data
num_train = len(train_dataset_aug)
indices = list(range(num_train))
split = int(0.9 * num_train)
random.seed(args.seed)
random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]

# 3. Create three distinct, disjoint subsets for each purpose
train_subset = Subset(train_dataset_aug, train_indices)         # For training, has augmentations
val_subset = Subset(train_dataset_no_aug, val_indices)          # For validation queries, no augmentations
gallery_subset = Subset(train_dataset_no_aug, train_indices)    # For validation gallery, no augmentations

# Final test set
test_dataset = CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

# 4. Create the final DataLoaders
train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
gallery_loader = DataLoader(gallery_subset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)


# In[ ]:


# =============================================================================
# DEFINE MODEL AND EVALUATE PRE-TRAINED ENCODER (UNTRAINED) BASELINE
# =============================================================================

model = MonoCon_ResNet(
    resnet_embedding_dim=args.resnet_embedding_dim,
    mlp_hidden_dim=args.mlp_hidden_dim,
    mlp_num_layers=args.mlp_num_layers,
    is_ablation=args.is_ablation,
).to(args.device)

print("\n--- Evaluating Pre-Trained ResNet Encoder (Baseline) on CIFAR-10 ---")
# For this initial check, we can use the full train set as the gallery
full_train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
y_train_initial, labels_train_initial = compute_embeddings(model, full_train_loader_no_aug, args.device, baseline_resnet_only=True)
y_test_initial, labels_test_initial = compute_embeddings(model, test_loader, args.device, baseline_resnet_only=True)
evaluate_embeddings(y_train_initial.numpy(), labels_train_initial.numpy(), y_test_initial.numpy(), labels_test_initial.numpy(), args.knn_k, "initial 'y' (pre-trained ResNet)")


# In[ ]:


# ================================================================================================
# EXECUTE HEAD WARMUP PHASE AND SET UP DIFFERENTIAL LEARNING RATE STRATEGY
# ================================================================================================

criterion = SupConLoss(temperature=args.temperature).to(args.device)

if not args.is_ablation:
    head_params = list(model.monotonic_mlp.parameters())
    if args.warmup_epochs > 0:
        print(f"\n--- Starting Head Warm-up for {args.warmup_epochs} epochs ---")
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer_warmup = torch.optim.AdamW(head_params, lr=args.lr_warmup, weight_decay=args.weight_decay)
        scheduler_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_warmup, T_max=args.warmup_epochs, eta_min=0)
        for epoch in range(args.warmup_epochs):
            epoch_desc = f"Warm-up Epoch {epoch+1}/{args.warmup_epochs}"
            avg_loss_warmup = run_epoch(model, train_loader, criterion, optimizer_warmup, args, epoch_desc, head_params)
            scheduler_warmup.step()
            print(f"{epoch_desc} -> Avg Loss: {avg_loss_warmup:.4f}")
        print("--- Warm-up finished. Unfreezing encoder for main training. ---")
        for param in model.encoder.parameters():
            param.requires_grad = True

    param_groups = [
        {'params': model.encoder.parameters(), 'lr': args.lr_resnet},
        {'params': head_params, 'lr': args.lr_mlp}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    print("--- Optimizer set up with differential learning rates for encoder and head for main training ---")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_resnet, weight_decay=args.weight_decay)
    print("--- Performing ablation study. Optimizer set up with the ResNet learning rate alone. ---")
    print("--- There is no warmup phase in the ablation study. ---")


# In[ ]:


# ================================================================================================
# MAIN TRAINING PHASE
# ================================================================================================
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
best_model_path = train_model(model, train_loader, val_loader, gallery_loader, criterion, optimizer, scheduler, args, output_dir)


# In[ ]:


# ================================================================================================
# FINAL EVALUATION PHASE
# ================================================================================================
print(f"\n--- Final Evaluation on Best Model ---")

if best_model_path and os.path.exists(best_model_path):
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    # For the final evaluation, we use the full training set as the gallery against the test set
    y_train_final, labels_train_final = compute_embeddings(model, full_train_loader_no_aug, args.device)
    y_test_final, labels_test_final = compute_embeddings(model, test_loader, args.device)

    y_train_norm = F.normalize(y_train_final, p=2, dim=1).numpy()
    y_test_norm = F.normalize(y_test_final, p=2, dim=1).numpy()

    model_name = "MonoCon Ablation" if args.is_ablation else "MonoCon"
    evaluate_embeddings(y_train_norm, labels_train_final.numpy(), y_test_norm, labels_test_final.numpy(), args.knn_k, f"final 'y' ({model_name})")

    visualize_embeddings(y_test_norm, labels_test_final.numpy(), output_dir, args.seed, f"y_final_{model_name}")

    print(f"\n--- PCA on Final 'y' Embeddings ---")
    pca = PCA(n_components=None)
    pca.fit(y_test_norm) # It's common to fit PCA on test embeddings for analysis of the final representation
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components_99_var = np.searchsorted(cumulative_variance, 0.99) + 1

    print(f"Number of components required to explain 99% of variance: {num_components_99_var}")
    if num_components_99_var <= len(cumulative_variance):
        print(f"   (The top {num_components_99_var} components actually explain {cumulative_variance[num_components_99_var-1] * 100:.2f}% of the variance)")

    print(f"\n--- Detailed breakdown for the first 20 components ---")
    for i in range(20):
        if i < len(pca.explained_variance_ratio_):
            var = pca.explained_variance_ratio_[i]
            cum_var = cumulative_variance[i]
            print(f"  - PC {i+1}: {var * 100:.2f}% (Cumulative: {cum_var * 100:.2f}%)")
else:
    print("Could not find a saved model. Skipping final evaluation.")


# In[ ]:




