from main.common.utils import get_relative_path

MODEL_NAME = 'dsl_lstm.model'
DATASET_NAME = 'dsl_dataset'
MODEL_DIR = get_relative_path(__file__, f'./algorithm/models/{DATASET_NAME}/{MODEL_NAME}/')