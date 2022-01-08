from tensorboard import program
import os


tb = program.TensorBoard()

DATASET_NAME = 'dsl_dataset'
LOG_DIR = f'./main/core/algorithm/models/{DATASET_NAME}/logs'

os.system(f'tensorboard --logdir={LOG_DIR}')
