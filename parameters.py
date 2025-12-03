from pathlib import Path
import torch 

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
TRAIN_PATH = DATA_DIR / "s00.csv"
VAL_PATH   = DATA_DIR / "s03.csv"
TEST_PATH  = DATA_DIR / "s02.csv"

# output paths
TRAIN_NORM_SAVE = "/Users/minwoongyoon/Documents/T-GAN/Data/train_data.csv"

# HMM
HMM_NUM_STATES = 3
HMM_N_ITER     = 40
HMM_COV_TYPE   = "diag"

# Tumbling window
TUMBLE_WIN_SIZE = 100    # you used 100

# GAN hyperparameters
SEQ_LEN   = 30
NOISE_DIM = 32
BATCH_SIZE = 32

GEN_SIZES  = [256, 64]
DISC_SIZES = [256, 64, 32]

LR_G = 2e-4
LR_D = 2e-3
BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4
LAMBDA_MSE = 0.05
EPOCHS = 100

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
