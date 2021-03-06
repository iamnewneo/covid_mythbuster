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
        "BATCH_SIZE": 16,
        "MAX_EPOCHS": 50,
        "LR": 3e-4,
        "N_WORKER": 16,
        "LABEL_PRED": {"LR_BASE": 1e-4, "LR_LINEAR": 1e-4},
        "RATIONALE_PRED": {"LR_BASE": 1e-4, "LR_LINEAR": 1e-3},
    },
}
SEED = 42
CONFIG = configs[ENV]
BASE_PATH = os.getenv("BASE_PATH", ".")

# Trainer
N_GPU = 4
N_WORKER = configs[ENV]["N_WORKER"]
FP_PRECISION = 16

# Training
DEVICE = configs[ENV]["DEVICE"]
BATCH_SIZE = configs[ENV]["BATCH_SIZE"]
MAX_EPOCHS = configs[ENV]["MAX_EPOCHS"]
RESUME_TRAINING = False

# Model
RATIONALE_MODEL_PATH = "./models/rationale_roberta_large_model/"
LABEL_MODEL_PATH = "./models/label_roberta_large_model/"
