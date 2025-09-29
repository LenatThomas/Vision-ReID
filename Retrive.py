import tkinter as tk
from tkinter import filedialog, ttk
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image, ImageTk
import torch
from config import Config as C
from Data.Gallery import SearchGallery
from Model.Vit import VIT
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def visualize_results(results, root, topk=5):
    results = results[:topk]

    ncols = min(topk, 5)  
    nrows = (len(results) + ncols - 1) // ncols

    plt.figure(figsize=(4*ncols, 6*nrows))

    for i, res in enumerate(results):
        meta = res["metadata"]
        imname = meta["imname"]
        box = meta["box"]  
        score = res["score"]

        img_path = root / imname
        img = Image.open(img_path).convert("RGB")

        if isinstance(box, torch.Tensor):
            box = box.tolist()
        x, y, w, h = map(float, box)
        box_xy = [x, y, x + w, y + h]

        draw = ImageDraw.Draw(img)
        draw.rectangle(box_xy, outline="red", width=3)

        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img)
        plt.title(f"Rank {i+1} | Score={score:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()




Source = C.SOURCE
modelPath = Source / 'models/VIT 1.1.pt'
galleryPath = Source / 'gallery/gallery_sysu0.pth'
root = Source / "datasets/cuhksysu/cuhksysu/"
imageDir = root / "Image/SSM/" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(modelPath, map_location=device)
gallery = SearchGallery(model = model, device = device)
gallery.load(galleryPath)

transform = Compose([
            Resize((256, 128)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

queryPath = Source / 'datasets/market1501/query/1268_c2s3_018782_00.jpg' 

query = Image.open(queryPath).convert("RGB")
query = transform(query)
results = gallery.search(query = query)
visualize_results(root = imageDir, results = results)

