import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, lr_scheduler
from torch import nn 
from torchinfo import summary
from VT import VIT
from Dataset import ReIDset

EPOCHS = 50
BATCHSIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(f'Using device {device}')
    generator = torch.Generator().manual_seed(0)
    transform = Compose([
        Resize((256, 128)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ReIDset(directory = 'reid/market/bounding_box_train' , transform = transform)
    trainSize = int(0.8 * len(dataset))
    valSize   = len(dataset) - trainSize
    train , val = random_split(dataset, [trainSize , valSize], generator = generator)
    trainLoader = DataLoader(dataset = train, batch_size = BATCHSIZE , shuffle = True, num_workers = 4)
    valLoader   = DataLoader(dataset = val , batch_size = BATCHSIZE , shuffle = False, num_workers = 4)
    model = VIT(imageHeight = 256, imageWidth = 128 , nClasses = dataset.nClasses).to(device = device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr = 3e-4, weight_decay = 0.05)
    bestAccuracy = 0.0
    print('Initialization done')
    for epoch in range(EPOCHS):
        print(f'Epoch : {epoch}')
        model.train()
        runningLoss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(trainLoader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs , labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = runningLoss / total
        train_acc = correct / total
    
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for images, labels in tqdm(valLoader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
    
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
    
        val_loss /= total
        val_acc = correct / total
    
        print(f"\nEpoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
        if val_acc > bestAccuracy:
            bestAccuracy = val_acc
            torch.save(model.state_dict(), "model.pth")
            print("✅ Best model saved.\n")




