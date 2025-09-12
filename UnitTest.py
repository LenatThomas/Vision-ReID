import os 
import torch
from pathlib import Path
from Model.Vit import VIT
from dotenv import load_dotenv
from Data.Dataset import CuhkSysuSearchSet
from Utils.Logger import setupLogger
from Data.Losses import mineTriplets
from Data.Sampler import PKBatchExpander
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, Normalize
from config import Config as C
from Data.Gallery import Gallery


VERSION = C.VERSION
BATCHSIZE = 128
logFile = C.logFile
logFile.parent.mkdir(exist_ok=True)
root    = C.root
split = C.split
logger = setupLogger(logFile = logFile)

transform = Compose([
            Resize((256, 128)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset     = CuhkSysuSearchSet(root = root, split = split ,  transform = transform)
for crop, imname, box in dataset:
    print(imname, box)
