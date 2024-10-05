import os
import shutil
import torch

from config import *
from model import BiGRU
from dataset import DataModule

from ray.tune import ExperimentAnalysis
from lightning.pytorch.trainer.trainer import Trainer

project_path = os.path.abspath('..')


def main(run_name):
    # To satisfy Lightning
    torch.set_float32_matmul_precision('medium')
    
    experiment_path = os.path.join('file://', project_path, 'results', run_name)
    analysis = ExperimentAnalysis(experiment_path)
    best_trial = analysis.get_best_trial(
        metric='val_f1',
        mode='max'
    )
    best_checkpoint = analysis.get_best_checkpoint(
        best_trial,
        metric='val_f1',
        mode='max'
    )
    best_config = analysis.get_best_config(
        metric='val_f1',
        mode='max'
    )['train_loop_config']
    best_checkpoint_path = os.path.join(best_checkpoint.path, 'checkpoint.ckpt')

    datamodule = DataModule(
        best_config,
        project_path
    )
    
    # A workaround to calculate class ratio and pass it to the model
    # so a loss function can level it out
    datamodule.setup('test')
    model = BiGRU.load_from_checkpoint(
        best_checkpoint_path,
        config=best_config,
        mean_target_length=datamodule.mean_target_length
    )
    trainer = Trainer(
        max_epochs=-1,
        devices=1,
        logger=False,
        enable_progress_bar=False
    )
    
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main(RUN_NAME)
