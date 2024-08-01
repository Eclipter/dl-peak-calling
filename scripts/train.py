import torch
from config import config
from model import BiGRU
from dataset import DataModule
from callbacks import TimeLogger
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer.trainer import Trainer
from ray.tune.integration.pytorch_lightning import TuneReportCallback

torch.set_float32_matmul_precision('medium')


def main():
    logger = CSVLogger(
        save_dir='results',
        name=None,
        version=config['RUN_NAME']
    )
    model = BiGRU(
        dataset_prefix=config['DATASET_PREFIX'],
        hidden_size=config['HIDDEN_SIZE'],
        num_layers=config['NUM_LAYERS'],
        lr=config['LR'],
        weight_decay=config['WEIGHT_DECAY'],
        loss_fn=config['LOSS_FN'],
        optimizer=config['OPTIMIZER'],
        dataset_number=config['DATASET_NUMBER'],
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        num_epochs=config['NUM_EPOCHS']
    )
    data_module = DataModule(
        dataset_number=config['DATASET_NUMBER'],
        dataset_prefix=config['DATASET_PREFIX'],
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS']
    )
    metrics = {'loss': 'val_loss', 'f1': 'val_f1'}
    trainer = Trainer(
        precision='16-mixed',
        max_epochs=config['NUM_EPOCHS'],
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                min_delta=1e-2,
                patience=5
            ),
            RichProgressBar(),
            TimeLogger(logger)
        ],
        enable_model_summary=False
    )
    
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
