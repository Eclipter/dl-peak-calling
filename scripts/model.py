import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn, optim
from lightning.pytorch.core import LightningModule


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


class BiGRU(LightningModule):
    def __init__(
        self,
        dataset_prefix,
        hidden_size,
        num_layers,
        lr,
        weight_decay,
        loss_fn,
        optimizer,
        dataset_number,
        batch_size,
        num_workers,
        num_epochs
        ):
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
        ) # (batch_size, length, hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1) # (batch_size, 200)
        
        self.loss_fn = eval(loss_fn)()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        
        self.dataset_number = dataset_number
        self.dataset_prefix = dataset_prefix
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        
        self.save_hyperparameters()
        
        self.f1_scorer = torchmetrics.F1Score(task='multilabel', num_labels=200)
        
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_targets, self.val_targets, self.test_targets = [], [], []
    
    def forward(self, x):
        bigru_output, _ = self.bigru(x)
        logits = self.fc(bigru_output).squeeze()
        
        return logits
    
    def _common_step(self, batch, stage):
        X_batch, y_batch = batch

        logits = self.forward(X_batch)
        y_proba = F.sigmoid(logits)
        
        loss = self.loss_fn(logits, y_batch)
        eval(f'self.{stage}_loss').append(loss)
        
        y_pred = (y_proba > 0.5).int()
        eval(f'self.{stage}_preds').append(y_proba)
        eval(f'self.{stage}_targets').append(y_batch)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, 'test')

    def _on_common_epoch_end(self, stage):
        loss = torch.stack(eval(f'self.{stage}_loss')).mean(dim=0)
        
        preds = torch.cat(eval(f'self.{stage}_preds'), dim=0)
        targets = torch.cat(eval(f'self.{stage}_targets'), dim=0)
        
        f1 = self.f1_scorer(preds, targets)

        self.log_dict(
            {
                f'{stage}_loss': loss,
                f'{stage}_f1': f1
            },
            sync_dist=True
        )

    def on_train_epoch_end(self):
        self._on_common_epoch_end('train')

    def on_validation_epoch_end(self):
        self._on_common_epoch_end('val')

    def on_test_epoch_end(self):
        self._on_common_epoch_end('test')
    
    def configure_optimizers(self):
        return eval(self.optimizer)(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
