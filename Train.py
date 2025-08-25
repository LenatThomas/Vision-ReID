import torch
import torch.nn as nn
import platform
import os
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch import nn 
from Model.Vit import VIT
from Data.Dataset import ReIDset
from Data.Sampler import PKBatchExpander
from Data.Gallery import Gallery
from Data.Losses import mineTriplets
from Utils.Logger import setupLogger
from Utils.Tracker import TrainingTracker
from torch.amp import GradScaler, autocast
from config import Config as C

load_dotenv()

VERSION         = C.VERSION
EPOCHS          = C.EPOCHS
LEARNING_RATE   = C.LEARNING_RATE
PATIENCE        = C.PATIENCE
CLIP            = C.CLIP
RESUME          = C.RESUME
P, K, BATCHSIZE = C.P, C.K, C.BATCHSIZE
logFile         = C.logFile
modelFile       = C.modelFile
dataPath        = C.dataPath
galleryPath     = C.galleryPath
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logFile.parent.mkdir(exist_ok=True)
modelFile.parent.mkdir(exist_ok = True)


logger = setupLogger(logFile = logFile)
logger.info(f"System: {platform.platform()}")
logger.info(f"Python Version: {platform.python_version()}")
logger.info(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA Available: {True}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
logger.info(f"Using device {device.type}")
logger.info(f"Using model {VERSION}")
logger.info(f"Hyperparameters: EPOCHS = {EPOCHS}, BATCHSIZE = {BATCHSIZE}, P = {P}, K = {K}, LR = {LEARNING_RATE}")


if __name__ == '__main__':

    try: 
        generator = torch.Generator().manual_seed(0)
        transform = Compose([
            Resize((256, 128)),
            ToTensor(),
            RandomHorizontalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset     = ReIDset(directory = dataPath , transform = transform)
        trainSize   = int(0.8 * len(dataset))
        valSize     = len(dataset) - trainSize
        train , val = random_split(dataset, [trainSize , valSize], generator = generator)
        trainLoader = DataLoader(dataset = train, num_workers = 4, batch_size = P, shuffle = True)
        valLoader   = DataLoader(dataset = val, num_workers = 4, batch_size = P, shuffle = True)
        expander    = PKBatchExpander(dataset = dataset)

        model       = VIT(imageHeight = 256, imageWidth = 128 , nClasses = dataset.nClasses).to(device = device)
        classificationCriterion = nn.CrossEntropyLoss()
        verificationCriterion   = nn.TripletMarginLoss(margin = 0.3) 
        optimizer               = AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = 0.05)
        logger.info(f"Criterion = [{classificationCriterion.__class__.__name__}, {verificationCriterion.__class__.__name__}] Optimizer = {optimizer.__class__.__name__} Sampler = {expander.__class__.__name__}")
        if len(trainLoader) == 0 or len(valLoader) == 0:
            logger.error("Empty DataLoader!")
            raise ValueError("DataLoader is empty")
        
        logger.info("Initialization Done")


        bestAccuracy = 0.0
        NoImprovements = 0
        scaler = GradScaler(device = device)
        trainTracker = TrainingTracker()
        valTracker   = TrainingTracker()

        if RESUME and modelFile.exists():
            checkpoint = torch.load(modelFile, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            bestAccuracy = checkpoint.get('bestAccuracy', 0.0)
            startEpoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"Resumed from checkpoint at epoch {startEpoch} with best accuracy {bestAccuracy:.4f}")
        else:
            startEpoch = 0

        startTime = datetime.now()
        logger.info('Training Started')
        for epoch in range(startEpoch, EPOCHS):
            logger.info(f'Epoch : {epoch}')
            model.train()
            trainTracker.reset()
            valTracker.reset()
            for images, labels, indices in tqdm(trainLoader, desc = "Training"):
                images = images.to(device)
                labels = labels.to(device)
                expandedImages, expandedLabels = expander.sample(labels = labels, indices = indices)
                expandedImages = expandedImages.to(device)
                expandedLabels = expandedLabels.to(device)
                optimizer.zero_grad()

                with autocast(device_type = device.type):
                    outputs, anchors = model(images)
                    _ , pools        = model(expandedImages)
                    triplets = mineTriplets(anchors = anchors, anchorLabels = labels, pools = pools, poolLabels = expandedLabels)
                    if triplets:
                        anchors     = triplets['anchors']
                        positives   = triplets['positives']
                        negatives   = triplets['negatives']
                        tLoss       = verificationCriterion(anchors, positives, negatives)
                    else : 
                        tLoss = torch.tensor(0.0, device=device)
                    cLoss   = classificationCriterion(outputs , labels)
                    loss    = cLoss + tLoss
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
                _, predicted = torch.max(outputs, 1)
                trainTracker.update(cLoss, batch = images.size(0) , predicted = predicted , labels = labels)


            model.eval()
            with torch.no_grad():
                for images, labels, indices in tqdm(valLoader, desc="Validating"):
                    images, labels = images.to(device), labels.to(device)
                    outputs, embeddings = model(images)
                    cLoss = classificationCriterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    valTracker.update(cLoss, batch = images.size(0), predicted = predicted, labels = labels)

            logger.info(f'Epoch {epoch} Train Loss {trainTracker.loss:.6f} Train Accuracy {trainTracker.accuracy:.6f} Validation Loss {valTracker.loss:.6f} Validation Accuracy {valTracker.accuracy:.6f}')
            
            if valTracker.accuracy > bestAccuracy:
                bestAccuracy = valTracker.accuracy
                NoImprovements = 0
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'bestAccuracy': bestAccuracy
                }, modelFile, pickle_protocol = 4)
                logger.info("Accuracy Improved, checkpoint saved")
            else:
                NoImprovements += 1
                logger.info(f"No improvement for {NoImprovements} epochs")
                if NoImprovements >= PATIENCE:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        logger.info('Training Ended')
        logger.info(f'Training Duration = {datetime.now() - startTime}')
        logger.info('Building Gallery')
        gallery = Gallery
        gallery.build(model = model, dataset = dataset)
        gallery.save(galleryPath)
        logger.info(f'Gallery Saved to {galleryPath}')

    except KeyboardInterrupt:
        logger.warning('Training Interrupted by user')
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        raise
    finally :
        logger.info('Pipeline Ended')




