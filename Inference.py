import torch
import os
import torch.nn as nn
from dotenv import load_dotenv
from pathlib import Path
from Model.Vit import VIT
from Data.Dataset import ReIDset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

VERSION = 'VIT 0.1'
load_dotenv()
dataPath = Path(os.getenv('SAVE_PATH')) / 'datasets/market1501/query/'
modelPath = Path(os.getenv('SAVE_PATH')) / f'models/{VERSION}.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = torch.Generator().manual_seed(0)
transform = Compose([
    Resize((256, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
testSet = ReIDset(dataPath, transform=transform)
testLoader = DataLoader(dataset=testSet, num_workers=4, batch_size=32, shuffle=False)
model = VIT(imageHeight=256, imageWidth=128, nClasses=testSet.nClasses + 1).to(device)
modelState = torch.load(modelPath, map_location=device)
model.load_state_dict(modelState)
model.eval()
criterion = nn.CrossEntropyLoss()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(testLoader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
avg_loss = val_loss / total
accuracy = 100 * correct / total
print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
