import pandas as pd
import os

internal_dyad_positions_path = os.path.join(
    os.path.abspath('..'),
    'data',
    'dataset_1',
    'cache',
    'internal_dyad_positions.txt'
)
internal_dyad_positions = pd.read_csv(
    internal_dyad_positions_path,
    header=None
)

a = internal_dyad_positions.map(lambda x: x[1:-1].split(', '))

# print(a.loc[137221].item())

templates_path = os.path.join(
    os.path.abspath('..'),
    'data',
    f'dataset_1',
    'cache',
    'templates.txt'
)
with open(templates_path) as file:
    templates = file.readlines()

print(templates[137221])