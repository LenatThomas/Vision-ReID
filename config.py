from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    VERSION = "Sysu 0.0"
    SOURCE = Path(os.getenv("SOURCE", "./outputs"))

    EPOCHS = 80
    LEARNING_RATE = 3e-4
    PATIENCE = 5
    CLIP = 1.0
    RESUME = False
    P = 128
    K = 6
    BATCHSIZE = P * K

    logFile     = SOURCE / f"logs/{VERSION}.txt"
    modelPath   = SOURCE / f"models/{VERSION}.pth"
    root        = SOURCE / "datasets/cuhksysu/cuhksysu/"
    galleryPath = SOURCE / f"gallery/gallery_{VERSION}.pth"
    split       = 'Train.mat'
    
