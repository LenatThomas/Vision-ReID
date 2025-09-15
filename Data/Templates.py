import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

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