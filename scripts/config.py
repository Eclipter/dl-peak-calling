# config = {
#     # Model hyperparameters
#     'HIDDEN_SIZE': [32, 64, 128, 256],
#     'NUM_LAYERS': [1, 2, 3],
#     'LOSS_FN': 'nn.BCEWithLogitsLoss',
#     'OPTIMIZER': 'optim.Adam',
#     'LR': [1e-3, 1e-4, 1e-5],
#     'WEIGHT_DECAY': [1e-4, 1e-5, 1e-6],

#     # Dataset hyperparameters
#     'DATASET_NUMBER': [1, 2, 3],
#     'DATASET_PREFIX': ['default', 'bigram', 'trigram'],
#     'BATCH_SIZE': 1024,
#     'NUM_WORKERS': 16,

#     # Training hyperparameters
#     'NUM_EPOCHS': 100,
#     'RUN_NAME': 'test'
# }
config = {
    # Model hyperparameters
    'HIDDEN_SIZE': 32,
    'NUM_LAYERS': 1,
    'LOSS_FN': 'nn.BCEWithLogitsLoss',
    'OPTIMIZER': 'optim.Adam',
    'LR': 1e-3,
    'WEIGHT_DECAY': 1e-6,

    # Dataset hyperparameters
    'DATASET_NUMBER': 1,
    'DATASET_PREFIX': 'trigram',
    'BATCH_SIZE': 1024,
    'NUM_WORKERS': 16,

    # Training hyperparameters
    'NUM_EPOCHS': 100,
    'RUN_NAME': 'test'
}
