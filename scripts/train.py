import os
import shutil
import torch

from config import *
from model import BiGRU
from dataset import DataModule

from lightning.pytorch.trainer.trainer import Trainer

from ray import init
from ray.tune import Tuner, TuneConfig
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import RayTrainReportCallback

project_path = os.path.abspath('..')

# A workaround which is relevant for me only
init(num_cpus=1, _temp_dir=os.path.join(os.path.expanduser('~'), '_scratch', 'tmp'))


def train_fn(config):
    # To satisfy Lightning
    torch.set_float32_matmul_precision('medium')
    
    datamodule = DataModule(
        config,
        project_path
    )
    
    # A workaround to calculate class ratio and pass it to the model
    # so a loss function can level it out
    datamodule.setup('train')
    model = BiGRU(
        config=config,
        mean_target_length=datamodule.mean_target_length
    )
    trainer = Trainer(
        max_epochs=1000,
        precision='16-mixed',
        callbacks=[RayTrainReportCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    
    trainer.fit(model, datamodule)


def custom_trial_name_creator(trial):
    trial_id = trial.trial_id.split('_')[-1]
    
    return f'trial_{trial_id}'


def main(config):
    # Remove a results folder with currently used name if exists
    shutil.rmtree(
        os.path.join(project_path, 'results', RUN_NAME),
        ignore_errors=True
    )
    
    
    # Configure Ray Tune
    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=True
    )
    checkpoint_config = CheckpointConfig(
        # num_to_keep=5,
        checkpoint_score_attribute='val_f1',
        checkpoint_score_order='max'
    )
    run_config = RunConfig(
        checkpoint_config=checkpoint_config,
        name=RUN_NAME,
        storage_path=os.path.join(project_path, 'results')
    )
    ray_trainer = TorchTrainer(
        train_fn,
        scaling_config=scaling_config,
        run_config=run_config
    )
    tune_config = TuneConfig(
        trial_dirname_creator=custom_trial_name_creator
    )
    tuner = Tuner(
        trainable=ray_trainer,
        param_space={'train_loop_config': config},
        tune_config=tune_config
    )
    
    tuner.fit()

if __name__ == '__main__':
    main(config)
