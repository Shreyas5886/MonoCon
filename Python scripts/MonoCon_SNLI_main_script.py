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

####################### Create a separate class to define all parameters #######################
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
    patience = 10                # Number of epochs after which early stopping is triggered
    batch_size = 128            # Mini-batch size
    warmup_epochs = 1           # Number of epochs to warm up the head. Set to 0 to disable.
    lr_warmup = 1e-4            # Learning rate specifically for the warm-up phase.
    lr_lm = 2e-4                # Learning rate for fine-tuning the language model encoder
    lr_mlp = 2e-4               # Learning rate for the monotonic MLP head 
    weight_decay = 1e-4         # Weight decay (L2) regularization
    temperature = 0.05          # Temperature parameter for SupCon loss. Controls the importance of hard negatives in the learning task
    grad_clip_norm = 1.0        # Max norm of gradients. Leads to more stable training
    seed = 42                   # random number generator seed
    is_ablation = False         # Set to True to bypass the monotonic MLP head for the ablation study training run.

    # System parameters
    base_results_dir = './results_SNLI' # Only the base path is needed here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Config()                  # Create an instance of the Config class to pass arguments to functions as needed
#################################################################################################


############### Set up the code for reproducibility ###################
print(f"Using device: {args.device}")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#######################################################################


############# Prepare output directories ##############################
# Create a unique, descriptive name for the experiment run
run_name = (
    f"temp_{args.temperature}_bs_{args.batch_size}_lr_encoder_{args.lr_lm}_epochs_{args.epochs}_patience_{args.patience}"
    f"{f'_lr_head_{args.lr_mlp}' if not args.is_ablation else ''}"  # <-- Conditionally add head learning rate
    f"{'_ablation' if args.is_ablation else ''}"
)

# Define the final output directory for this specific run
output_dir = os.path.join(args.base_results_dir, run_name)
os.makedirs(output_dir, exist_ok=True)

print(f"Output directory for this run: {output_dir}")
#######################################################################


# In[ ]:


# =============================================================================
# MODEL AND DATA CLASSES
# =============================================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss function."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device                                       # Identify the device on which the features tensor is stored
        batch_size = features.shape[0]                                 # Extract the batch_size. Useful for later operations
        labels = labels.contiguous().view(-1, 1)                       # Assign a continuous memory block to sotre labels. Reshape the labels tensor
        mask = torch.eq(labels, labels.T).float().to(device)           # Creates a square matrix mask via broadcasting, to identify positive pairs
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)  # Matrix of scaled feature dot products
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)            # Compute maximum value of the scaled dot product in each row
        # The next line subtracts a constant from each row, which leaves the loss unchanged but prevents numerical instability
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask                     # Removes diagonal elements, so that positive pairs of samples with themselves are not counted

        # ====== Main SupCon loss calculation =============
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        # =================================================
        return loss

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
# HELPER FUNCTIONS
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


def evaluate_encoder_only_stsb(encoder_model, tokenizer, stsb_data, args):
    """Evaluates the encoder part of the model only."""
    encoder_model.eval()
    device = args.device

    sents1 = [ex['sentence1'] for ex in stsb_data]
    sents2 = [ex['sentence2'] for ex in stsb_data]

    @torch.no_grad()
    def get_encoder_embeds(sentences):
        all_embeds = []
        pbar_embed = tqdm(range(0, len(sentences), args.batch_size), desc="Computing encoder embeddings", leave=False)
        for start_idx in pbar_embed:
            batch_sents = sentences[start_idx : start_idx + args.batch_size]
            tokenized = tokenizer(batch_sents, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt')
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            # Manually perform the encoder forward pass and pooling
            model_output = encoder_model(**tokenized)
            token_embeddings = model_output[0]
            input_mask_expanded = tokenized['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            all_embeds.append(pooled_embeddings.cpu())
        return torch.cat(all_embeds, dim=0)

    embeddings1 = get_encoder_embeds(sents1)
    embeddings2 = get_encoder_embeds(sents2)

    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    cosine_scores = torch.sum(embeddings1 * embeddings2, dim=1)
    labels = torch.tensor([ex['score'] for ex in stsb_data])
    spearman_corr, _ = spearmanr(cosine_scores.numpy(), labels.numpy())

    print(f"Spearman Correlation: {spearman_corr:.4f}")
    return spearman_corr


# In[ ]:


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def run_epoch(model, dataloader, criterion, optimizer, args, epoch_desc, params_to_clip):
    """Runs a single training epoch and returns the average loss."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=epoch_desc, unit="batch")

    for tokenized_batch, labels in pbar:
        # Make sure tokenized embeddings and lables are on the same device as other tensors
        tokenized_batch = {k: v.to(args.device) for k, v in tokenized_batch.items()}   
        labels = labels.to(args.device)

        optimizer.zero_grad()                                             # Zero out gradients from the previous iteration

        y_batch = model(**tokenized_batch)                   # Forward pass. Compute model output
        y_batch_norm = F.normalize(y_batch, p=2, dim=1)      # Normalize output embeddings

        loss = criterion(y_batch_norm, labels)               # Compute SupCon loss
        loss.backward()                                      # Calculate gradients (backward pass)

        # Clip gradients for the specified parameters
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=args.grad_clip_norm)   # Implement gradient norm clipping for stability
        optimizer.step()    # Update model weights

        total_loss += loss.item()                              # Add batch loss to total loss for the epoch
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})        # Attach postfix description to progress bar

    return total_loss / len(dataloader)                        # Return average loss


def train_model(model, train_loader, stsb_val_data, tokenizer, criterion, optimizer, scheduler, args, output_dir):
    """
    Main training loop with robust, fully resumable checkpointing.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the STSb validation set.
        tokenizer (Tokenizer): Generator of word tokens
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        args (Config): Configuration object with hyperparameters.
        output_dir (str): The directory to save checkpoints and the best model.

    Returns:
        str: The file path to the best performing model's weights, or None if no model was saved.
    """
    # --- 1. INITIALIZATION AND PATH DEFINITION ---
    start_epoch = 0
    best_val_corr = -1
    epochs_no_improve = 0

    # Define all file paths used in the function for clarity
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    latest_checkpoint_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    final_best_model_path = None # Will store the path of the best model if one is saved

    # --- 2. LOAD CHECKPOINT IF IT EXISTS ---
    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        best_val_corr = checkpoint['best_val_corr']
        # Load early stopping counter, defaulting to 0 if not found for backward compatibility
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0) 

        print(f"   Resumed from epoch {start_epoch}. Best validation loss: {best_val_corr:.4f}")
        print(f"   Patience counter (epochs with no improvement): {epochs_no_improve}")
    else:
        print("Starting training from scratch.")

    # --- 3. MAIN TRAINING LOOP ---
    for epoch in range(start_epoch, args.epochs):
        epoch_desc = f"Epoch {epoch+1}/{args.epochs} (Train)"
        avg_train_loss = run_epoch(model, train_loader, criterion, optimizer, args, epoch_desc, model.parameters())
        val_corr = evaluate_stsb(model, tokenizer, stsb_val_data, args, silent=True)
        scheduler.step()

        # Print epoch summary
        # First, determine the learning rate string based on the mode
        if not args.is_ablation:
            lr_log_str = f"LRs (MiniLM/Head): {optimizer.param_groups[0]['lr']:.1e}/{optimizer.param_groups[1]['lr']:.1e}"
        else:
            lr_log_str = f"LR (MiniLM): {optimizer.param_groups[0]['lr']:.1e}"

        # Then, use that string in a single, clean print statement
        print(f"Epoch {epoch+1}/{args.epochs} -> "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Spearman Corr: {val_corr:.4f} | "
              f"{lr_log_str}")

        # --- 4. SAVE/COPY OPERATIONS & EARLY STOPPING ---

        # A) Save the best model if validation loss improves
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            final_best_model_path = best_model_path # Update the final path
            print(f"New best model saved with Val Correlation: {best_val_corr:.4f}")
        else:
            epochs_no_improve += 1

        # B) Save a comprehensive checkpoint for the current epoch
        current_epoch_checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_corr': best_val_corr,
            'epochs_no_improve': epochs_no_improve,
        }, current_epoch_checkpoint_path)

        # C) Update the 'latest' checkpoint to point to this epoch's file
        shutil.copyfile(current_epoch_checkpoint_path, latest_checkpoint_path)

        # D) Check for early stopping
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs with no improvement.")
            break

    print("--- Training Finished ---")
    return final_best_model_path


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


# In[ ]:


# =============================================================================
# DEFINE MODEL AND EVALUATE PRE-TRAINED ENCODER BASELINE
# =============================================================================

model = MonoCon_LM(
    model_name=args.model_name,
    lm_embedding_dim=args.lm_embedding_dim,
    mlp_hidden_dim=args.mlp_hidden_dim,
    mlp_num_layers=args.mlp_num_layers,
    is_ablation=args.is_ablation,
).to(args.device)

print("\n--- Evaluating Pre-Trained MiniLM Encoder (Baseline) on STSb ---")
evaluate_encoder_only_stsb(model.encoder, tokenizer, stsb_test_data, args)


# In[ ]:


# ================================================================================================
# EXECUTE HEAD WARMUP PHASE AND SET UP DIFFERENTIAL LEARNING RATE STRATEGY IF RUNNING FULL MODEL
# ================================================================================================

criterion = SupConLoss(temperature=args.temperature).to(args.device)    # Define SupCon loss as the loss function

if not args.is_ablation:

    head_params = list(model.monotonic_mlp.parameters()) # Define head parameters separately for warmup phase and differential learning rate strategy

    if args.warmup_epochs > 0:      # The warmup phase is executed only if number of warmup epochs is a positive integer
        print(f"\n--- Starting Head Warm-up for {args.warmup_epochs} epochs ---")

        # Freeze the encoder layers
        for param in model.encoder.parameters():
            param.requires_grad = False

        # Create a new optimizer and scheduler just for the head during warm-up
        # The 'head_params' variable is already defined above
        optimizer_warmup = torch.optim.AdamW(head_params, lr=args.lr_warmup, weight_decay=args.weight_decay)   # Optimizer is set up with learning rate for the head warmup phase
        scheduler_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_warmup, T_max=args.warmup_epochs, eta_min=0)

        # Simplified training loop for the warm-up phase
        for epoch in range(args.warmup_epochs):

            epoch_desc = f"Warm-up Epoch {epoch+1}/{args.warmup_epochs}"
            avg_loss_warmup = run_epoch(model, train_loader, criterion, optimizer_warmup, args, epoch_desc, head_params)
            scheduler_warmup.step()      # Update learning rate according to the warmup learning rate scheduler
            print(f"{epoch_desc} -> Avg Loss: {avg_loss_warmup:.4f}")

        # Unfreeze the encoder layers for the main training phase
        print("--- Warm-up finished. Unfreezing encoder for main training. ---")
        for param in model.encoder.parameters():
            param.requires_grad = True

    # Set up optimizer for main training
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': args.lr_lm},      # Parameters and learning rate for encoder
        {'params': head_params, 'lr': args.lr_mlp}                         # Parameters and learning rate for monotonic MLP head  
    ]
    # Note: 'params' and 'lr' are special dictionary keys recognized by the AdamW optimizer, and 'params' is mandatory. 
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)   # Set up AdamW optimizer with differential learning rates for encoder and head
    print("--- Optimizer set up with differential learning rates for encoder and head for main training ---")

else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_lm, weight_decay=args.weight_decay) # No differential learning rates for the ablation study
    print("--- Performing ablation study. Optimizer set up with the MiniLM learning rate alone. ---")
    print("--- There is no warmup phase in the ablation study. ---")


# In[ ]:


# ================================================================================================
# MAIN TRAINING PHASE
# ================================================================================================
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
best_model_path = train_model(model, train_loader, stsb_val_data, tokenizer, criterion, optimizer, scheduler, args, output_dir)


# In[ ]:


# ================================================================================================
# FINAL EVALUATION PHASE
# ================================================================================================
print(f"\n--- Final Evaluation on Best Model ---")

# Check if the model was successfully trained and saved
if best_model_path and os.path.exists(best_model_path):
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    evaluate_stsb(model, tokenizer, stsb_test_data, args)

    print("--- Starting Final Analysis: PCA and Reconstruction Error ---")

 # =============================================================================
    # 1. HELPER FUNCTION TO GENERATE EMBEDDINGS
    # =============================================================================
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

    # =============================================================================
    # 2. GENERATE AND NORMALIZE EMBEDDINGS FOR TRAIN AND TEST SETS
    # =============================================================================
    # NOTE: This script assumes 'model' is the best-performing model, already loaded.
    model.eval()

    # --- Generate Test Embeddings (STSb) ---
    print("\nStep 1: Generating embeddings for the TEST dataset (STSb)...")
    sents1_test = [ex['sentence1'] for ex in stsb_test_data]
    sents2_test = [ex['sentence2'] for ex in stsb_test_data]
    unique_test_sentences = list(set(sents1_test + sents2_test))
    test_embeddings_tensor = generate_embeddings(unique_test_sentences, model, tokenizer, args)
    # Normalize embeddings for PCA, as it's sensitive to vector magnitude
    test_embed = F.normalize(test_embeddings_tensor, p=2, dim=1).numpy()
    print(f"Generated {test_embed.shape[0]} unique test embeddings of dimension {test_embed.shape[1]}.")

    # --- Generate Train Embeddings (SNLI) ---
    print("\nStep 2: Generating embeddings for the TRAIN dataset (SNLI)...")
    train_sentences = list(set([s for p, h in train_dataset.samples for s in (p, h)]))
    train_embeddings_tensor = generate_embeddings(train_sentences, model, tokenizer, args)
    # Normalize embeddings
    train_embed = F.normalize(train_embeddings_tensor, p=2, dim=1).numpy()
    print(f"Generated {train_embed.shape[0]} unique train embeddings of dimension {train_embed.shape[1]}.")

    # =============================================================================
    # 3. PERFORM PCA AND RECONSTRUCTION ERROR ANALYSIS
    # =============================================================================
    print("\nStep 3: Performing PCA and calculating generalization error...")

    # --- Fit a full PCA on the training data to find the optimal number of components ---
    print("\n--- Fitting PCA on training data to analyze variance ---")
    pca_full = PCA(n_components=None, random_state=42)
    pca_full.fit(train_embed)

    # Find the number of components that explain 99% of the variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_99_variance = np.searchsorted(cumulative_variance, 0.99) + 1
    print(f"Number of components to explain 99% of TRAIN variance: {n_for_99_variance}")

    # --- Calculate Reconstruction Error to Quantify Generalization ---
    print(f"\n--- Calculating Test Set Reconstruction Error using top {n_for_99_variance} components from train subspace ---")

    # 1. Define a new PCA model with the determined number of components
    pca_reconstruction = PCA(n_components=n_for_99_variance, random_state=42)

    # 2. Fit this PCA model ONLY on the training data to define the subspace
    pca_reconstruction.fit(train_embed)

    # 3. Reconstruct the test data by projecting it onto the train subspace and then inverting
    test_projected = pca_reconstruction.transform(test_embed)
    test_reconstructed = pca_reconstruction.inverse_transform(test_projected)

    # 4. Calculate the Mean Squared Error between the original and reconstructed test data
    reconstruction_mse = np.mean(np.square(test_embed - test_reconstructed))

    # 5. Calculate the Root Mean Squared Error between original and reconstructed test data
    reconstruction_rmse = np.sqrt(np.mean(np.square(test_embed - test_reconstructed)))
    print(f"Test Set Reconstruction RMSE from Train Subspace: {reconstruction_rmse:.8f}")

    print(f"Test Set Reconstruction MSE from Train Subspace: {reconstruction_mse:.8f}")

    print("\n--- Analysis Complete ---")
else:
    print("Could not find a saved model. Skipping final evaluation.")

