from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    VERSION = "VIT 1.1"
    SOURCE = Path(os.getenv("SOURCE", "./outputs"))

    EPOCHS = 80
    LEARNING_RATE = 3e-4
    PATIENCE = 20
    CLIP = 1.0
    RESUME = False
    P = 128
    K = 6
    BATCHSIZE = P * K

    logFile     = SOURCE / f"logs/{VERSION}.txt"
    modelFile   = SOURCE / f"models/{VERSION}.pth"
    dataPath    = SOURCE / "datasets/market1501/bounding_box_train/"
    galleryPath = SOURCE / "gallery/gallery.pth"
    queryPath   = SOURCE / "datasets/market1501/query/"
