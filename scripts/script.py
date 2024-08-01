import os
import pickle
from time import time, gmtime, strftime
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
from Bio import SeqIO

import matplotlib.pyplot as plt
from matplotlib.image import imread
# Uncomment to use dark plots
# plt.style.use('dark_background')
import seaborn as sns

from IPython.display import clear_output
import nbformat
# Automatically choose import based on the file type
if 'get_ipython' in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

SKYBLUE = '#B3E6FF'
INDIGO = '#4B0082'
LAVENDER = '#F0CCFF'
MAGENTA = '#D166FF'


# To manually set a device specify it instead of 'None' (if necessary)
device = torch.device('cuda:1')

# Automatically set a better device
if not device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up a hyperparameter for distributed training
if device.type == 'cuda':
    world_size = torch.cuda.device_count()


class DefaultDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

        self.ohe = {
            'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
            'C': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
            'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
            'T': torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        dyad_positions = self.targets[idx]

        # Encode sequence
        encoded_seq = torch.stack([self.ohe[nuc] for nuc in seq])

        # Encode dyad position
        encoded_dyad_positions = torch.zeros(200)
        encoded_dyad_positions[dyad_positions] = 1

        return encoded_seq, encoded_dyad_positions


class BigramDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

        self.nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        dyad_positions = self.targets[idx] - 1

        label_encoded_seq = [self.nuc_to_idx[nuc] for nuc in seq]

        # Create bi-grams
        bigrams = []
        for i in range(len(label_encoded_seq) - 1):
            bigrams.append((label_encoded_seq[i], label_encoded_seq[i+1]))

        # One-hot encode sequence
        one_hot_encoded_seq = []
        for bigram in bigrams:
            bigram_idx = bigram[0] * 16 + bigram[1] * 4 + bigram[2]
            one_hot_encoded_seq.append(np.eye(16)[bigram_idx])
        one_hot_encoded_seq = torch.tensor(np.array(one_hot_encoded_seq), dtype=torch.float32)

        # Encode dyad position
        encoded_dyad_positions = torch.zeros(200)
        encoded_dyad_positions[dyad_positions] = 1

        return one_hot_encoded_seq, encoded_dyad_positions


class TrigramDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

        self.nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        dyad_positions = self.targets[idx] - 2

        label_encoded_seq = [self.nuc_to_idx[nuc] for nuc in seq]

        # Create tri-grams
        trigrams = []
        for i in range(len(label_encoded_seq) - 2):
            trigrams.append((label_encoded_seq[i:i+3]))

        # One-hot encode sequence
        one_hot_encoded_seq = []
        for trigram in trigrams:
            c
            one_hot_encoded_seq.append(np.eye(64)[trigram_idx])
        one_hot_encoded_seq = torch.tensor(np.array(one_hot_encoded_seq), dtype=torch.float32)
        
        # Encode dyad position
        encoded_dyad_positions = torch.zeros(200)
        encoded_dyad_positions[dyad_positions] = 1

        return one_hot_encoded_seq, encoded_dyad_positions


def load_templates(dataset_number):
    with open(f'../dataset_{dataset_number}/cache/templates.pickle', 'rb') as file:
        templates = pickle.load(file)
    
    return templates


def load_internal_dyad_positions(dataset_number):
    with open(f'../dataset_{dataset_number}/cache/internal_dyad_positions.pickle', 'rb') as file:
        internal_dyad_positions = pickle.load(file)
    
    return internal_dyad_positions


def create_dataset(
    templates,
    internal_dyad_positions,
    dataset_prefix
    ):
    if dataset_prefix == 'default':
        return DefaultDataset(templates, internal_dyad_positions)
    
    elif dataset_prefix == 'bigram':
        return BigramDataset(templates, internal_dyad_positions)
    
    elif dataset_prefix == 'trigram':
        return TrigramDataset(templates, internal_dyad_positions)


def split_dataset(dataset):
    # indices = np.random.choice(range(len(dataset)), 100000)
    # dataset = torch.utils.data.Subset(dataset, indices)
    
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    datasets = {}
    datasets['train'], datasets['val'], datasets['test'] = random_split(dataset, [train_size, val_size, test_size])
    
    return datasets


def create_dataloaders(split_dataset, batch_size, samplers):
    dataloader = {}
    if samplers == None:
        dataloader['train'] = DataLoader(
            split_dataset['train'],
            batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        dataloader['val'] = DataLoader(
            split_dataset['val'],
            batch_size,
            drop_last=True,
            pin_memory=True
        )
        dataloader['test'] = DataLoader(
            split_dataset['test'],
            batch_size,
            drop_last=True,
            pin_memory=True
        )
        
    else:
        dataloader['train'] = DataLoader(
            split_dataset['train'],
            batch_size,
            drop_last=True,
            pin_memory=True,
            sampler=samplers['train']
        )
        dataloader['val'] = DataLoader(
            split_dataset['val'],
            batch_size,
            drop_last=True,
            pin_memory=True,
            sampler=samplers['val']
        )
        dataloader['test'] = DataLoader(
            split_dataset['test'],
            batch_size,
            drop_last=True,
            pin_memory=True,
            sampler=samplers['test']
        )
    
    return dataloader


class MulticlassBiGRU(nn.Module):
    def __init__(self, dataset_prefix, hidden_size, num_layers):
        super().__init__()
        
        ngram_symbol_num = {
            'default': 1,
            'bigram': 2,
            'trigram': 3
            
        }
        
        self.bigru = nn.GRU(
            input_size=4**ngram_symbol_num[dataset_prefix],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def __repr__(self):
        return 'MulticlassBiGRU'

    def forward(self, x):
        bigru_output, _ = self.bigru(x) # (batch_size, length, hidden_size * 2)
        fc_output = self.fc(bigru_output).squeeze() # (batch_size, length)
        y_proba = self.softmax(fc_output) # (batch_size, length)
        
        return y_proba


class MultilabelBiGRU(nn.Module):
    def __init__(self, dataset_prefix, hidden_size, num_layers):
        super().__init__()
        
        ngram_symbol_num = {
            'default': 1,
            'bigram': 2,
            'trigram': 3
            
        }
        
        self.bigru = nn.GRU(
            input_size=4**ngram_symbol_num[dataset_prefix],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def __repr__(self):
        return 'MultilabelBiGRU'

    def forward(self, x):
        bigru_output, _ = self.bigru(x) # (batch_size, length, hidden_size * 2)
        fc_output = self.fc(bigru_output).squeeze() # (batch_size, length)     
        y_proba = self.sigmoid(fc_output) # (batch_size, length)
        
        return y_proba


def train(
    rank=0,
    world_size=1,
    distributed=False,
    args=dict()
    ):
    model = args['model']
    model_str = f"{args['model']}"
    hidden_size = args['hidden_size']
    num_layers = args['num_layers']
    loss_fn = args['loss_fn']
    optimizer = args['optimizer']
    lr = args['lr']
    weight_decay = args['weight_decay']
    num_epochs = args['num_epochs']
    batch_size = args['batch_size']
    dataset_prefix = args['dataset_prefix']
    dataset_number = args['dataset_number']
    run_name = args['run_name']
    
    start_time = time()
    
    # Set up distributed processes
    if distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '50000'
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    else:
        rank = device

    # Initialize objects
    model = model(
        dataset_prefix,
        hidden_size,
        num_layers
    ).to(rank)
    if distributed:
        model = DDP(model)
    loss_fn = loss_fn()#weight=torch.tensor([0.1, 0.9]))
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataloaders = prepare_dataloaders(
        dataset_prefix,
        dataset_number,
        batch_size,
        world_size,
        rank,
        distributed
    )

    train_history = {'loss': [], 'f1': []}
    val_history = {'loss': [], 'f1': []}
    if distributed and rank == 0:
        epoch_pbar = tqdm(
            total=num_epochs,
            desc='Epochs',
            colour=SKYBLUE,
            leave=False
        )
    for epoch in range(1, num_epochs + 1):
        

        model, loss, f1, pred_dists['train'] = train_epoch(
            model,
            hidden_size,
            num_layers,
            loss_fn,
            optimizer,
            lr,
            weight_decay,
            num_epochs,
            batch_size,
            start_time,
            epoch,
            rank,
            distributed,
            dataloaders['train'],
            pred_dists['train']
        )
        train_history['loss'].append(loss)
        train_history['f1'].append(f1)

        loss, f1, pred_dists['val'] = evaluate(
            model,
            loss_fn,
            num_epochs,
            batch_size,
            epoch,
            rank,
            distributed,
            dataloaders['val'],
            pred_dists['val']
        )
        val_history['loss'].append(loss)
        val_history['f1'].append(f1)

        plot_dashboard(
            model_str,
            hidden_size,
            num_layers,
            loss_fn,
            optimizer,
            lr,
            weight_decay,
            num_epochs,
            batch_size,
            dataset_prefix,
            dataset_number,
            run_name,
            start_time,
            epoch,
            rank,
            distributed,
            train_history,
            val_history,
            pred_dists
        )
        
        if distributed and rank == 0:
            epoch_pbar.update(1)

    # Save the final results
    if rank == 0:
        results = {
            'model_state': model.module.state_dict(),
            'train_history': train_history,
            'val_history': val_history,
            'pred_dists': pred_dists
        }
        torch.save(results, f'../results/{run_name}/results.pth')
    
    if distributed:
        dist.destroy_process_group()


def train_epoch(
        model,
        hidden_size,
        num_layers,
        loss_fn,
        optimizer,
        lr,
        weight_decay,
        num_epochs,
        batch_size,
        start_time,
        epoch,
        rank,
        distributed,
        dataloader,
        pred_dists
    ):
    model.train()
    
    if distributed and rank == 0:
        step_pbar = tqdm(
            total=len(dataloader),
            desc=f'      Training {epoch}/{num_epochs}',
            colour=INDIGO,
            leave=False
        )
    batch_loss, batch_preds, batch_targets = [], [], []
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(rank), y_batch.to(rank)

        y_proba = model(X_batch)

        # Collect predictions and targets
        y_pred = (y_proba > 0.3).int()
        
        batch_preds.append(y_pred.detach().cpu())
        batch_targets.append(y_batch.detach().cpu())

        # Calculate loss
        if f'{loss_fn}' in ['WeightedBCELoss', 'WeightedMSELoss']:
            loss = loss_fn(y_proba, y_batch, batch_size)
        else:
            loss = loss_fn(y_proba, y_batch)
        batch_loss.append(loss.item())

        # Get predicted dyad distributions
        dyad_indices = torch.argwhere(y_pred == 1)[:, 1].tolist()
        pred_dists += Counter(dyad_indices)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if distributed and rank == 0:
            step_pbar.update(1)
    
    # Get loss values averaged across all batches
    loss = np.mean(batch_loss)
    
    # Calculate F1 score averaged across all batches
    batch_preds = torch.cat(batch_preds, dim=0)
    batch_targets = torch.cat(batch_targets, dim=0)
    
    f1 = f1_score(
        batch_preds,
        batch_targets,
        average='weighted',
        zero_division=0
    )
    
    return model, loss, f1, pred_dists


def evaluate(
    model,
    loss_fn,
    num_epochs,
    batch_size,
    epoch,
    rank,
    distributed,
    dataloader,
    pred_dists
    ):
    model.eval()
    
    if distributed and rank == 0:
        step_pbar = tqdm(
            total=len(dataloader),
            desc=f'    Evaluating {epoch}/{num_epochs}',
            colour=LAVENDER,
            leave=False
        )
    batch_loss, batch_preds, batch_targets = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(rank), y_batch.to(rank)
            
            # Get predictions
            y_proba = model(X_batch)
            
            # Collect predictions and targets
            y_pred = (y_proba > 0.3).int()
            
            batch_preds.append(y_pred.detach().cpu())
            batch_targets.append(y_batch.detach().cpu())
            
            # Calculate loss
            if f'{loss_fn}' in ['WeightedBCELoss', 'WeightedMSELoss']:
                loss = loss_fn(y_proba, y_batch, batch_size)
            else:
                loss = loss_fn(y_proba, y_batch)
            batch_loss.append(loss.item())
            
            # Get predicted dyad distributions
            dyad_indices = torch.argwhere(y_pred == 1)[:, 1].tolist()
            pred_dists += Counter(dyad_indices)
            
            if distributed and rank == 0:
                step_pbar.update(1)
    
    # Get loss values averaged across all batches
    loss = np.mean(batch_loss)
    
    # Calculate F1 score averaged across all batches
    batch_preds = torch.cat(batch_preds, dim=0)
    batch_targets = torch.cat(batch_targets, dim=0)
    
    f1 = f1_score(
        batch_preds,
        batch_targets,
        average='weighted',
        zero_division=0
    )
    
    return loss, f1, pred_dists


def plot_dashboard(
    model,
    hidden_size,
    num_layers,
    loss_fn,
    optimizer,
    lr,
    weight_decay,
    num_epochs,
    batch_size,
    dataset_prefix,
    dataset_number,
    run_name,
    start_time,
    epoch,
    rank,
    distributed,
    train_history,
    val_history,
    pred_dists
    ):
    # Set up axes
    fig = plt.figure(figsize=(20, 10))
    axs = fig.subplot_mosaic(
        [['top_left', 'top_right'],
         ['bottom_left', 'bottom_right']]
    )
    
    # Plot the loss
    axs['top_left'].plot(
        train_history['loss'],
        color=INDIGO,
        linewidth=3,
        label='Train'
    )
    axs['top_left'].plot(
        val_history['loss'],
        color=MAGENTA,
        linewidth=3,
        label='Validation'
    )
    axs['top_left'].set_xlabel('Epochs')
    axs['top_left'].set_ylabel('Loss')
    axs['top_left'].legend()
    axs['top_left'].spines[['right', 'top']].set_visible(False)
    axs['top_left'].grid(False)
    
    # Plot the F1 score
    axs['top_right'].plot(
        train_history['f1'],
        color=INDIGO,
        linewidth=3,
        label='Train'
    )
    axs['top_right'].plot(
        val_history['f1'],
        color=MAGENTA,
        linewidth=3,
        label='Validation'
    )
    axs['top_right'].set_xlabel('Epochs')
    axs['top_right'].set_ylabel('F1 score')
    axs['top_right'].legend()
    axs['top_right'].spines[['right', 'top']].set_visible(False)
    axs['top_right'].grid(False) 

    # Plot the dyad probability distribution
    data = list(pred_dists['train'].elements())
    if len(data) == 0:
        data = np.random.randint(1, 201, 10)
    sns.histplot(
        data,
        bins=200,
        binwidth=1,
        binrange=(0, 201),
        stat='probability',
        color=INDIGO,
        edgecolor='black',
        linewidth=0.7,
        label='Train',
        ax=axs['bottom_left']
    )
    
    data = list(pred_dists['val'].elements())
    if len(data) == 0:
        data = np.random.randint(1, 201, 10)
    sns.histplot(
        data,
        bins=200,
        binwidth=1,
        binrange=(0, 201),
        stat='probability',
        color=MAGENTA,
        edgecolor='black',
        linewidth=0.7,
        label='Validation',
        ax=axs['bottom_left']
    )
    xticklabels = axs['bottom_left'].get_xticklabels()
    axs['bottom_left'].set_xticks(axs['bottom_left'].get_xticks()+0.5)
    axs['bottom_left'].set_xticklabels(xticklabels)
    axs['bottom_left'].set_xlabel('Nucleotide steps, bp')
    axs['bottom_left'].set_ylabel('Dyad probability')
    axs['bottom_left'].legend()
    axs['bottom_left'].spines[['right', 'top']].set_visible(False)
    axs['bottom_left'].grid(False)

    # Display information about hyperparameters
    model = str(model).split('(')[0]
    loss_fn = str(loss_fn).split('(')[0]
    optimizer = str(optimizer).split('(')[0]
    lr = f'{lr:e}'.replace('0', '').replace('.', '')
    weight_decay = f'{weight_decay:e}'.replace('0', '').replace('.', '')
    exec_time = round(time() - start_time)
    overall_time = strftime('%H:%M:%S', gmtime(exec_time))
    time_per_epoch = strftime('%H:%M:%S', gmtime(exec_time/epoch))


    axs['bottom_right'].text(
        x=0.2,
        y=-0.12,
        s=f'''
        • Model: {model}\n
        • Hidden size: {hidden_size}\n
        • Number of recurrent layers: {num_layers}\n
        • Loss function: {loss_fn}\n
        • Optimizer: {optimizer}\n
        • Learning rate: {lr}\n
        • Weight decay: {weight_decay}\n
        • Batch size: {batch_size}\n
        • Dataset: {dataset_prefix}_dataset_{dataset_number}\n\n
                    Time running\n
Epoch {epoch}/{num_epochs}, {overall_time} ({time_per_epoch} per epoch)
        ''',
        fontsize='15',
        linespacing = 0.7
    )
    axs['bottom_right'].set_title('Current Hyperparameters', fontsize=18, y=0.95)
    axs['bottom_right'].spines[['right', 'left', 'top', 'bottom']].set_visible(False)
    axs['bottom_right'].tick_params(
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )
    axs['bottom_right'].grid(False)

    # Save the dashboard
    if rank == 0:
        if not os.path.exists(f'../results/{run_name}/'):
            os.mkdir(f'../results/{run_name}/')
        plt.savefig(f'../results/{run_name}/dashboard.png', dpi=300, bbox_inches='tight')

    # Reload the dashboard after each epoch if in jupyter notebook
    elif not distributed:
        clear_output(wait=True)
        plt.show()


def show_dashboard(run_name):
    fig, ax = plt.subplots(figsize=(40, 25))

    dashboard = imread(f'../results/{run_name}/dashboard.png')

    plt.imshow(dashboard)

    plt.axis('off')

    plt.show()


class WeightedBCELoss():
    def __call__(self, y_proba, y_true, batch_size):
        # Calculate weights to penalize predictions quadratically
        indices = torch.arange(200, dtype=torch.float32).repeat(batch_size, 1).to(device)
        dyad_indices = y_true.unsqueeze(1).to(device)
        weights = F.mse_loss(indices, dyad_indices, reduction='none')

        # Calculate loss
        loss = F.binary_cross_entropy(y_proba, y_true)
        weighted_loss = (loss * weights).mean()

        return weighted_loss

    def __repr__(self):
        return 'WeightedBCELoss'


class WeightedMSELoss():
    def __call__(self, y_proba, y_true, batch_size):
        # Create tensors to caluclate loss between
        indices = torch.arange(200, dtype=torch.float32).repeat(batch_size, 1).to(device)
        dyad_indices = y_true.unsqueeze(1).to(device)

        # Calculate loss
        loss = F.mse_loss(indices, dyad_indices, reduction='none')
        weighted_loss = (loss * y_proba).mean()

        return weighted_loss

    def __repr__(self):
        return 'WeightedMSELoss'


def dist_train(world_size, args):
    distributed = True
    
    mp.spawn(
        train,
        args=(world_size, distributed, args),
        nprocs=world_size,
        join=True
    )


args = {
    'model': MultilabelBiGRU,
    'hidden_size': 512,
    'num_layers': 2,
    'loss_fn': nn.BCELoss,
    'optimizer': optim.Adam,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'num_epochs': 1,
    'batch_size': 512,
    'dataset_prefix': 'default',
    'dataset_number': 1,
    'run_name': '512_2_multilabel'
}


if __name__ == '__main__':
    if 'get_ipython' in globals():
        pass
    else:
        dist_train(world_size, args)


