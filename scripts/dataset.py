import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.core import LightningDataModule


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
        dyad_positions = np.array(self.targets[idx]) - 1

        label_encoded_seq = [self.nuc_to_idx[nuc] for nuc in seq]

        # Create bi-grams
        bigrams = []
        for i in range(199):
            bigrams.append((label_encoded_seq[i:i+2]))

        # One-hot encode sequence
        one_hot_encoded_seq = []
        for bigram in bigrams:
            bigram_idx = bigram[0] * 4 + bigram[1]
            one_hot_encoded_seq.append(np.eye(16)[bigram_idx])
        one_hot_encoded_seq = torch.tensor(np.array(one_hot_encoded_seq), dtype=torch.float32)

        # Encode dyad position
        encoded_dyad_positions = torch.zeros(199)
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
        dyad_positions = np.array(self.targets[idx]) - 2

        label_encoded_seq = [self.nuc_to_idx[nuc] for nuc in seq]

        # Create tri-grams
        trigrams = []
        for i in range(198):
            trigrams.append((label_encoded_seq[i:i+3]))

        # One-hot encode sequence
        one_hot_encoded_seq = []
        for trigram in trigrams:
            trigram_idx = trigram[0] * 16 + trigram[1] * 4 + trigram[2]
            one_hot_encoded_seq.append(np.eye(64)[trigram_idx])
        one_hot_encoded_seq = torch.tensor(np.array(one_hot_encoded_seq), dtype=torch.float32)
        
        # Encode dyad position
        encoded_dyad_positions = torch.zeros(198)
        encoded_dyad_positions[dyad_positions] = 1

        return one_hot_encoded_seq, encoded_dyad_positions


class DataModule(LightningDataModule):
    def __init__(self, config, project_path):
        super().__init__()
        
        self.dataset_number = config['DATASET_NUMBER']
        self.k = config['K']
        self.batch_size = config['BATCH_SIZE']
        self.num_workers = config['NUM_WORKERS']
        self.path = project_path

    def setup(self, stage):
        templates_path = os.path.join(
            self.path,
            'data',
            f'dataset_{self.dataset_number}',
            'cache',
            'templates.txt'
        )
        with open(templates_path) as file:
            templates = file.readlines()
        
        internal_dyad_positions_path = os.path.join(
            self.path,
            'data',
            f'dataset_{self.dataset_number}',
            'cache',
            'internal_dyad_positions.txt'
        )
        internal_dyad_positions = pd.read_csv(
            internal_dyad_positions_path,
            header=None
        )
        internal_dyad_positions = internal_dyad_positions.map(lambda x: x[1:-1].split(', '))
        
        self.mean_target_length = internal_dyad_positions.apply(len).mean()
        
        if self.k == 1:
            dataset = DefaultDataset(templates, internal_dyad_positions)
        elif self.k == 2:
            dataset = BigramDataset(templates, internal_dyad_positions)
        elif self.k == 3:
            dataset = TrigramDataset(templates, internal_dyad_positions)
        
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )
    
    def _common_dataloader(self, stage):
        return DataLoader(
            eval(f'self.{stage}_dataset'),
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers
        )
        
    def train_dataloader(self):
        return self._common_dataloader('train')

    def val_dataloader(self):
        return self._common_dataloader('val')

    def test_dataloader(self):
        return self._common_dataloader('test')
    
    def predict_dataloader(self):
        return self._common_dataloader('predict')
