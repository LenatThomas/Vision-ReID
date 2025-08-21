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

load_dotenv()

VERSION = 'Unit Test'
BATCHSIZE = 16
logFile = Path(os.getenv("SAVE_PATH"))/ f"logs/{VERSION}.txt"
logFile.parent.mkdir(exist_ok=True)

dataPath = Path(os.getenv("SAVE_PATH"))/ f"datasets/market1501/bounding_box_train/"
logger = setupLogger(logFile = logFile)

transform = Compose([
            Resize((256, 128)),
            ToTensor(),
            RandomHorizontalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

device = 'cpu'

generator = torch.Generator().manual_seed(0)
dataset = ReIDset(directory = dataPath, transform = transform)
trainSize   = int(0.8 * len(dataset))
valSize     = len(dataset) - trainSize
train , val = random_split(dataset, [trainSize , valSize], generator = generator)
loader = DataLoader(dataset = train, num_workers = 4, batch_size = BATCHSIZE)
expander = PKBatchExpander(dataset = dataset, k = 4)
model       = VIT(imageHeight = 256, imageWidth = 128 , nClasses = dataset.nClasses).to(device = device)
for images , labels, indices in loader:
    expandedBatch, expandedLabels = expander.sample(indices = indices, labels = labels)
    logger.info(indices)
    logger.info(labels)
    logger.info(expandedLabels)
    _, anchors = model(images)
    _, pools   = model(expandedBatch)
    triplets = mineTriplets(anchors = anchors, anchorLabels = labels , pools = pools, poolLabels = expandedLabels)
    logger.info(f"{triplets['positives'][0]}")


