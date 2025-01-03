# Training hyperparameters
INPUT_SIZE = 784
HIDDEN_SIZE = 50
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MIN_EPOCHS = 1
MAX_EPOCHS = 1000

# Dataset
DATADIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = 1
PRECISION = "16-mixed"
