from ray.tune import grid_search

RUN_NAME = 'test'

config = {
    # Model hyperparameters
    'HIDDEN_SIZE': 256,
    'NUM_LAYERS': 1,
    'LR': 1e-3,
    'WEIGHT_DECAY': 1e-6,

    # Dataset hyperparameters
    'DATASET_NUMBER': 1,
    'K': grid_search([2, 3]),
    'BATCH_SIZE': 2048,
    'NUM_WORKERS': 16,
}
