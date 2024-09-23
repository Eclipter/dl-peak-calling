import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn, optim
from lightning.pytorch.core import LightningModule


class BiGRU(LightningModule):
    def __init__(self, config, mean_target_length):
        super().__init__()
        
        self.bigru = nn.GRU(
            input_size=4**config['K'],
            hidden_size=config['HIDDEN_SIZE'],
            num_layers=config['NUM_LAYERS'],
            batch_first=True,
            bidirectional=True
        ) # (batch_size, length, hidden_size * 2)
        self.fc = nn.Linear(config['HIDDEN_SIZE'] * 2, 1) # (batch_size, length)
        
        pos_weight = torch.tensor((201-config['K']) / mean_target_length)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lr = config['LR']
        self.weight_decay = config['WEIGHT_DECAY']
        
        self.f1_scorer = torchmetrics.F1Score(
            task='multilabel',
            num_labels=201-config['K'],
            average='weighted'
        )
    
    def forward(self, x):
        bigru_output, _ = self.bigru(x)
        logits = self.fc(bigru_output).squeeze()
        
        return logits
    
    def _common_step(self, batch, stage):
        X_batch, y_batch = batch

        logits = self.forward(X_batch)
        y_proba = F.sigmoid(logits)
        
        loss = self.loss_fn(logits, y_batch)
        
        f1 = self.f1_scorer(y_proba, y_batch)
        
        self.log_dict(
            {
                f'{stage}_loss': loss,
                f'{stage}_f1': f1
            },
            on_epoch=True,
            on_step=False,
            sync_dist=True
        )
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, 'test')
    
    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
