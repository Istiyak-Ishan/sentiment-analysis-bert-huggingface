from pathlib import Path

# Model
MODEL_NAME = "bert-base-uncased"

# Dataset
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"

# Training
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
SEED = 42

# Output directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "saved_model"