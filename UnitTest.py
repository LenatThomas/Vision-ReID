import os 
import torch
from pathlib import Path
from Model.Vit import VIT
from dotenv import load_dotenv
from Data.Dataset import ReIDset
from Utils.Logger import setupLogger
from Data.Losses import mineTriplets
from Data.Sampler import PKBatchExpander
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, Normalize
from config import Config as C
from Data.Gallery import Gallery


modelFile = C.modelFile

load_dotenv()

VERSION = 'Unit Test'
BATCHSIZE = 128
logFile = Path(os.getenv("SAVE_PATH"))/ f"logs/{VERSION}.txt"
logFile.parent.mkdir(exist_ok=True)

dataPath = Path(os.getenv("SAVE_PATH"))/ f"datasets/market1501/bounding_box_train/"
savePath = Path(os.getenv("SAVE_PATH"))/ f"gallery/gallery.pth"
logger = setupLogger(logFile = logFile)

transform = Compose([
            Resize((256, 128)),
            ToTensor(),
            RandomHorizontalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ReIDset(directory = dataPath, transform = transform)
model       = VIT(imageHeight = 256, imageWidth = 128 , nClasses = dataset.nClasses).to(device = device)
checkpoint = torch.load(modelFile, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
gallery = Gallery(device = device)
gallery.build(model = model , dataset = dataset)
gallery.save(savePath)


