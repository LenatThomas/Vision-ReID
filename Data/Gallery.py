import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from Data.Templates import Gallery

class SearchGallery(Gallery):
    def build(self, dataset, batchsize = 128, numworker = 4):
        self._clear()
        loader = DataLoader(dataset = dataset, batch_size = batchsize, num_workers = numworker)
        features , metadata = [] , []
        with torch.no_grad():
            for crops, imname, box in tqdm(loader, desc = 'Building Gallery'):
                crops = crops.to(self._device)
                embeddings = self._model(crops)
                features.append(embeddings.cpu())    
                metadata.extend([{'imname' : imname[i], 'box' : box[i]} for i in range(len(imname))])
        self._features = torch.cat(features)
        self._metadata = metadata

class Gallery :
    def __init__(self, device = torch.device('cpu')):
        self._model = None
        self._device = device
        self._features = None
        self._labels = None
        self._filenames = None

    def isbuilt(self):
        return self._features is not None
    
    def _clear(self):
        self._features  = None
        self._labels    = None
        self._filenames = None
        
    def build(self, dataset , model):
        self._clear()
        self._model = model
        self._model.eval()
        loader = DataLoader(dataset = dataset, batch_size = 128, num_workers = 4)
        features, targets, filenames = [], [], []
        with torch.no_grad():
            for images, labels, indices in tqdm(loader, desc = 'Building Gallery'):
                images = images.to(self._device)
                embeddings = self._model(images)
                features.append(embeddings.cpu())
                targets.append(labels.cpu())
                for index in indices:
                    filenames.append(dataset.filename(index))
        self._features = torch.cat(features)
        self._labels = torch.cat(targets)
        self._filenames = filenames

    def save(self, filePath):
        if not self.isbuilt():
            raise RuntimeError("Gallery is empty, build or load first")
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        torch.save({
            "features": self._features,
            "labels": self._labels,
            "filenames": self._filenames
        }, filePath)

    def load(self, filePath):
        self._clear()
        checkpoint = torch.load(filePath, map_location="cpu")
        self._features = checkpoint["features"]
        self._labels = checkpoint["labels"]
        self._filenames = checkpoint["filenames"]

    def search(self, query , topK = 5):
        if not self.isbuilt():
            raise RuntimeError("Gallery is empty, build or load first")
        if self._model is None:
            raise RuntimeError("No model loaded in Gallery. Please set self._model before searching.")
        self._model.eval()
        if query.dim() == 3:
            query = query.unsqueeze(0)
        with torch.no_grad():
            query = query.to(self._device, non_blocking=True)
            embedding = self._model(query)
            embedding = F.normalize(embedding, p=2, dim=1)
         
        gallery = F.normalize(self._features.to(self._device), p=2, dim=1)
        similarity = torch.mm(query, gallery.t())
        scores, indices = torch.topk(similarity, k = topK , dim=1)

        results = []
        for b in range(query.size(0)):
            result = []
            for idx, score in zip(indices[b], scores[b]):
                result.append({
                    "filename": self._filenames[idx],
                    "label": int(self._labels[idx]),
                    "score": float(score)
                })
            results.append(result)
        return results

    def __len__(self):
        if not self.isbuilt():
            return 0
        return len(self._features)

    def __iter__(self):
        for i in range(len(self)):
            yield {
                "feature": self._features[i],
                "label": int(self._labels[i]),
                "filename": self._filenames[i]
            }

    def __getitem__(self, index):
        if not self.isbuilt():
            raise RuntimeError("Gallery is mepty, build or load first")
        if index >= len(self._features) or index < 0:
            raise IndexError("Index out of range")
        return {
            "feature": self._features[index],
            "label": int(self._labels[index]),
            "filename": self._filenames[index]
        }

