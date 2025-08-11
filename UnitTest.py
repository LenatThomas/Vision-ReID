import os 
import torch
from pathlib import Path
from dotenv import load_dotenv
from Dataset import ReIDset, PKSampler
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, Normalize

load_dotenv()
dataPath = Path(os.getenv("SAVE_PATH"))/ f"datasets/market1501/bounding_box_train/"

transform = Compose([
            Resize((256, 128)),
            ToTensor(),
            RandomHorizontalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
generator = torch.Generator().manual_seed(0)
dataset = ReIDset(directory = dataPath, transform = transform)
trainSize   = int(0.8 * len(dataset))
valSize     = len(dataset) - trainSize
train , val = random_split(dataset, [trainSize , valSize], generator = generator)
sampler = PKSampler(dataset = dataset , indices = train.indices, p = 20 , k = 6)
loader = DataLoader(dataset = dataset, batch_sampler = sampler)

for batch_num, batch_indices in enumerate(sampler):
    print(f"Batch {batch_num+1} indices:", batch_indices)
    labels = [dataset._labels[dataset._pids[i]] for i in batch_indices]
    print("Labels:", labels)
    print("-" * 40)