from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    VERSION = "VIT 1.1"
    SAVE_PATH = Path(os.getenv("SAVE_PATH", "./outputs"))

    EPOCHS = 80
    LEARNING_RATE = 3e-4
    PATIENCE = 20
    CLIP = 1.0
    RESUME = False
    P = 128
    K = 6
    BATCHSIZE = P * K

    logFile   = SAVE_PATH / f"logs/{VERSION}.txt"
    modelFile = SAVE_PATH / f"models/{VERSION}.pth"
    dataPath  = SAVE_PATH / "datasets/market1501/bounding_box_train/"
