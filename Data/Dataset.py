import os
import numpy as np
import torchvision.transforms as T
from PIL import Image
from Templates import ReidentificationDataset

class Market1501IdentificationSet(ReidentificationDataset):
    def __init__(self, root, transform = None):
        super().__init__(root, transform or T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        self._imageDir = os.path.join(self._root, "bounding_box_train")
        self._paths = [fname for fname in os.listdir(self._imageDir) if fname.lower().endswith(('.jpg', '.jpeg'))]
        self._length = len(self._paths)
        self._pids = [self._extractInfo(fname)[0] for fname in self._paths]
        self._cids = [self._extractInfo(fname)[1] for fname in self._paths]
        self._buildDict()

    def _extractInfo(self, fname):
        splits = fname.split('_')
        pid    = int(splits[0])
        cid    = splits[1]
        return (pid , cid)
    
    def _buildDict(self):
        self._indexMap = {}
        for index , pid in enumerate(self._pids):
            if pid not in self._indexMap:
                self._indexMap[pid] = []
            self._indexMap[pid].append(index)
    
    def __getitem__(self, index):
        if not (0 <= index < self._length):
            raise IndexError('Index out of range')
        fname = os.path.join(self._imageDir, self._paths[index])
        image = Image.open(fname).convert('RGB')
        if self._transform:
            image = self._transform(image)
        pid = self._pids[index]
        return image, pid, index
    
    def __len__(self):
        return self._length
    
    def filename(self, index):
        if not (0 <= index < self._length):
            raise IndexError('Index out of range')
        return self._paths[index]       