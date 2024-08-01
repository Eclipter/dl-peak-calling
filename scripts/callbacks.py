from time import time
from lightning.pytorch.callbacks import Callback


class TimeLogger(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time()

    def on_train_epoch_end(self, trainer, pl_module):
        exec_time = int(time() - self.start_time)
        
        self.logger.log_metrics({'exec_time': exec_time})
