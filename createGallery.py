import torch
import os
from Data.Gallery import SearchGallery
from Data.Dataset import CuhkSysuSearchSet
from config import Config as C

Source = C.SOURCE
modelPath = Source / 'models/VIT 1.1.pt'
galleryPath = Source / 'gallery/gallery_sysu0.pth'
root = Source/ "datasets/cuhksysu/cuhksysu/"
device = torch.device('cuda')

print(f'Using device {device}')
dataset = CuhkSysuSearchSet(root = root)
model = torch.load(modelPath, map_location=device)
gallery = SearchGallery()
gallery.build(dataset = dataset, model = model , device = device, batchsize = 128, numworker = 4)
gallery.save(filepath = galleryPath)
print(f'Gallery saved to {galleryPath}')
