import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

class ReidentificationDataset(Dataset, ABC):
    def __init__(self, root ,transform = None):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root {root} does not exist")
        self._root = root
        self._transform = transform
        self._pids = []
        self._indexMap = {}
        self._iterator = 0
        self._length = 0
        self._pid2Label = {}
        self._label2Pid = {}

    @abstractmethod
    def _buildDict(self):
        """Build the pid -> Index mapping"""
        pass

    @abstractmethod
    def __getitem__(self, index):
        """ Return (image, pid, index)"""
        pass

    def __len__(self):
        return self._length
    
    def _buildMap(self):
        uniquePids = sorted(set(self._pids))
        self._pid2Label = {pid : label for label, pid in enumerate(uniquePids)}
        self._label2Pid = {label : pid for pid, label in self._pid2Label.items()}

    @abstractmethod
    def filename(self, index):
        """Return filename for the given index"""
        pass

    @property
    def indexMap(self):
        return self._indexMap
    
    @property
    def pids(self):
        return self._pids
    
    @property
    def nClasses(self):
        return len(set(self._pids))
    
    @property
    def labelMap(self):
        return self._pid2Label
    
    @property
    def pidMap(self):
        return self._label2Pid
    
    def __iter__(self):
        self._iterator = 0
        return self
    
    def __next__(self):
        if self._iterator < len(self):
            item = self.__getitem__(self._iterator)
            self._iterator += 1
            return item[0] , item[1]
        else :
            raise StopIteration

    def __str__(self):
        return (f"{self.__class__.__name__} (root={self._root}, images={len(self)}, identities={self.nClasses})")
    
    def __repr__(self):
        return str(self)
    
class SearchDataset(Dataset, ABC):
    def __init__(self, root , transform):
        self._root = root
        self._transform = transform
        self._cropMaps = {}
        self._length = 0
        self._iterator = 0

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return self._length
    
    @abstractmethod
    def _buildMapping(self):
        pass

    @property
    def nImages(self):
        return self.nImages
    
    @property
    def cropMaps(self):
        return self._cropMaps
    
    def __iter__(self):
        self._iterator = 0
        return self
    
    def __next__(self):
        if self._iterator <= len(self):
            crops, imname, boxes = self.__getitem__(self._iterator)
            self._iterator += 1
            return crops, imname, boxes
        else :
            raise StopIteration
        
    def __str__(self):
        return f'{self.__class__.__name__} ({len(self)})'
    
    def __repr__(self):
        return str(self)
    
class Gallery(Dataset, ABC):
    def __init__(self, model, device = torch.device('cpu')):
        super().__init__()
        self._device = device
        self._model = model
        self._model.eval()
        self._features = None
        self._metadata = None

    def isbuilt(self):
        return self._features is not None and self._metadata is not None \
           and len(self._features) == len(self._metadata) \
           and len(self._features) > 0
    
    def _clear(self):
        self._features = None
        self._metadata = None

    @abstractmethod
    def build(self, dataset):
        pass

    def save(self, filepath):
        if not self.isbuilt():
            raise RuntimeError('Gallery is empty, build or load first')
        directory = os.path.dirname(filepath)
        if directory != '':
            os.makedirs(directory, exist_ok = True)
        torch.save({
            'features' : self._features,
            'metadata' : self._metadata
        }, filepath)

    def load(self, filepath):
        self._clear()
        checkpoint = torch.load(filepath, map_location = 'cpu')
        self._features = checkpoint['features']
        self._metadata = checkpoint['metadata']

    def search(self, query, topk = 5):
        if not self.isbuilt():
            raise RuntimeError('Gallery is empty, build or load first')
        if self._model is None:
            raise RuntimeError("No model loaded in Gallery. Please set self._model before searching.")
        self._model.eval()
        if query.dim() == 3:
            query = query.unsqueeze(0)
        with torch.no_grad():
            query = query.to(self._device, non_blocking=True)
            embedding = self._model(query)
            embedding = F.normalize(embedding, p=2, dim=1)

        gallery = F.normalize(self._features.to(self._device), p = 2 , dim = 1)
        gallery = gallery.cpu()
        similarity = torch.mm(embedding.cpu(), gallery.t())
        scores, indices = torch.topk(similarity, k = topk, dim = 1)

        results = []
        for b in range(query.size(0)):
            result = []
            for idx, score in zip(indices[b], scores[b]):
                result.append({
                    "metadata" : self._metadata[idx],
                    "score" : float(score)
                })
            results.append(result)

        if len(results) == 1:
            return results[0]
        return results

    def __len__(self):
        if not self.isbuilt():
            return 0
        return len(self._features)
    
    def __iter__(self):
        for i in range(len(self)):
            yield {
                "feature" : self._features[i],
                "metadata" : self._metadata[i]
            }

    def __getitem__(self, index):
        if not self.isbuilt():
            raise RuntimeError('Gallery is empty, build or load first')
        if not (0 <= index < len(self._features)): 
            raise IndexError("Index out of range")
        return {
            'feature' : self._features[index],
            'metadata' : self._metadata[index]
        }