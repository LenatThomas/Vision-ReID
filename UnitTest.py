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
from Model.Vit import VIT, VITEmbedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelPath = C.modelPath
checkPointPath = modelPath.with_suffix('.pth')
embeddingPath = modelPath.with_suffix('.pt')
checkpoint = torch.load(checkPointPath, map_location=device)
bestModel = VIT(imageHeight=256, imageWidth=128, nClasses=751).to(device)
bestModel.load_state_dict(checkpoint['model_state'])
embeddingModel = VITEmbedding(baseModel = bestModel)
torch.save(embeddingModel, embeddingPath)
