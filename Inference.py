import torch
import os
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import random_split
from pathlib import Path
from Model.Vit import VIT
from Data.Dataset import ReIDset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config as C
from Data.Gallery import Gallery
from Utils.Logger import setupLogger
from Utils.Metrics import mAP, rankK

VERSION = 'VIT 1.1'
load_dotenv()
SOURCE = Path(os.getenv("SOURCE", "./outputs"))
logFile = SOURCE / f"logs/inference.txt"
logger = setupLogger(logFile = logFile)

queryPath   = C.queryPath
modelPath   = C.modelFile
galleryPath = C.galleryPath

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('cpu')
transform = Compose([
    Resize((256, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

logger.info(f"Using device {device.type}")
logger.info(f"Using model {VERSION}")

query       = ReIDset(queryPath, transform=transform)
trainSize   = int(0.9 * len(query))
valSize     = len(query) - trainSize
train , val = random_split(query, [trainSize , valSize])
        
loader      = DataLoader(dataset = val, num_workers = 4, batch_size = 128, shuffle = False)
gallery     = Gallery()
gallery.load(galleryPath)
model       = VIT(imageHeight = 256, imageWidth = 128, nClasses = query.nClasses + 1).to(device)
checkpoint  = torch.load(modelPath, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.to(device)

logger.info(f"Using model {model.__class__.__name__}")

results   = []
targets   = []
predicted = []
for images, labels, indices in tqdm(loader, desc = 'Testing...'):
    images = images.to(device)
    _, embeddings = model(images)
    batchResults = gallery.search(embeddings, topK = 10)
    results.extend(batchResults)
    for i in batchResults:
        predicted.append([k['label'] for k in i])
    targets.extend([k.item() for k in labels])


ks = [1 , 2 , 3 , 5 , 10]
precision = mAP(predicted, targets)
accuracy  = rankK(predicted, targets , ks = ks)
logger.info(f"Mean Average Precision: {precision}")
for k in ks:
    logger.info(f"Rank-{k} Accuracy : {accuracy[k]}")






