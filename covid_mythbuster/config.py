import os

ENV = os.getenv("ENV", "dev")
configs = {
    "dev": {
        "DEVICE": "cpu",
        "BATCH_SIZE": 2,
        "MAX_EPOCHS": 2,
        "LR": 1,
        "N_WORKER": 0,
        "LABEL_PRED": {"LR_BASE": 1, "LR_LINEAR": 1},
        "RATIONALE_PRED": {"LR_BASE": 1, "LR_LINEAR": 1},
    },
    "prod": {
        "DEVICE": "cuda",
        "BATCH_SIZE": 128,
        "MAX_EPOCHS": 500,
        "LR": 3e-4,
        "N_WORKER": 16,
        "LABEL_PRED": {"LR": 1},
        "RATIONALE_PRED": {"LR": 1},
    },
}
SEED = 42
CONFIG = configs[ENV]
BASE_PATH = os.getenv("BASE_PATH", ".")

# Trainer
N_GPU = 8
N_WORKER = configs[ENV]["N_WORKER"]
FP_PRECISION = 16

# Training
DEVICE = configs[ENV]["DEVICE"]
BATCH_SIZE = configs[ENV]["BATCH_SIZE"]
MAX_EPOCHS = configs[ENV]["MAX_EPOCHS"]
RESUME_TRAINING = False

# Model
