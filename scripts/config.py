from ray.tune import grid_search

RUN_NAME = 'final_search'

config = {
    # Model hyperparameters
    'HIDDEN_SIZE': 256,
    'NUM_LAYERS': 1,
    'LR': 1e-3,
    'WEIGHT_DECAY': 1e-6,

    # Dataset hyperparameters
    'DATASET_NUMBER': 1,
    'DATASET_PREFIX': grid_search(['bigram', 'trigram']),
    'BATCH_SIZE': 2048,
    'NUM_WORKERS': 16,
}
