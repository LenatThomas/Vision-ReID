import os
import numpy as np
import torchvision.transforms as T
from PIL import Image

class ReIDset():
    def __init__(self, directory, transform = None):
        if not os.path.exists(directory):
            raise FileNotFoundError("Provided path doesn't exist")
        self._directory = directory
        self._paths = [fname for fname in os.listdir(directory) if fname.endswith(('.jpg', '.jpeg'))]
        self._length = len(self._paths)
        self._pids = [self._extractInfo(fname)[0] for fname in self._paths]
        self._cids = [self._extractInfo(fname)[1] for fname in self._paths]
        self._transform = transform or T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._iterator = 0
        self._dict = {}
        self._buildDict()

    def _extractInfo(self, fname):
        splits = fname.split('_')
        pid    = int(splits[0])
        cid    = splits[1]
        return (pid , cid)
    
    def _buildDict(self):
        self._dict = {}
        for index , pid in enumerate(self._pids):
            if pid not in self._dict.keys():
                self._dict[pid] = []
            self._dict[pid].append(index)
    
    def __getitem__(self, index):
        if index >= self._length or index < 0:
            raise IndexError('Index out of range')
        fname = os.path.join(self._directory, self._paths[index])
        image = Image.open(fname).convert('RGB')
        image = self._transform(image)
        pid = self._pids[index]
        return image, pid, index
    
    def __len__(self):
        return self._length
    
    def __iter__(self):
        self._iterator = 0
        return self
    
    def __next__(self):
        index = self._iterator
        self._iterator += 1
        if self._iterator < self._length:
            fname = os.path.join(self._directory, self._paths[index])
            image = Image.open(fname).convert('RGB')
            image = self._transform(image)
            pid = self._pids[index]
            return image, pid
        else :
            raise StopIteration

    def fileName(self, index):
        if index >= self._length or index < 0:
            raise IndexError('Index out of range')
        return self._paths[index]       
        
    @property
    def pids(self):
        return self._pids
        
    @property
    def indexMap(self):
        return self._dict

    @property
    def nClasses(self):
        return len(self._dict.keys())
