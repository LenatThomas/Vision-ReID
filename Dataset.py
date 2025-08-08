import os
import random
import torchvision.transforms as T
import torch.utils.data.sampler as Sampler 
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
        self._labels = {pid : idx for idx, pid in enumerate(set(self._pids))}
        self._transform = transform or T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._iterator = 0

    def _extractInfo(self, fname):
        splits = fname.split('_')
        pid    = splits[0]
        cid    = splits[1]
        return (pid , cid)
    
    def __getitem__(self, index):
        if index >= self._length or index < 0:
            raise IndexError('Index out of range')
        fname = os.path.join(self._directory, self._paths[index])
        image = Image.open(fname).convert('RGB')
        image = self._transform(image)
        pid = self._pids[index]
        label = self._labels[pid]
        return image, label
    
    def __len__(self):
        return self._length
    
    def __iter__(self):
        self._interator = 0
        return self
    
    def __next__(self):
        index = self._interator
        self._interator += 1
        if self._iterator < self._length:
            fname = os.path.join(self._directory, self._paths[index])
            image = Image.open(fname).convert('RGB')
            image = self._transform(image)
            pid = self._pids[index]
            label = self._labels[pid]
            return image, label
        else :
            raise StopIteration

    @property
    def nClasses(self):
        return len(self._labels.keys())

class PKSampler(Sampler):
    def __init__(self, dataset, batchSize, nInstances):
        self._pidDict = {}
        self._batches = []
        self._pids    = []
        self._data = dataset
        self._batchSize = batchSize
        self._nInstances = nInstances
        self._build()

        
    def _prepareBatches(self):
        self._pids = random.shuffle(self._pids)
        batch = []
        while(self.pid):
            for i in self._pids:
                nsamples = min(self._nInstances , len(self._pidDict[i]))
                random.shuffle(self._pidDict[i])
                samples = self._pidDict[i][0:nsamples]
                self._pidDict[i] = self._pidDict[i][nsamples:]
                
                for k in samples:
                    batch.append(k)
                    if len(batch) > self._batchSize:
                        self._batches.append(batch)
                        batch = []



    def _build(self):
        self._pidDict = {}
        for index, (_, label) in enumerate(self._data):
            if label not in self._pidDict.keys():
                self._pidDict[label] = [index]
            else :
                self._pidDict[label].append(index)
        self._pids = self._pidDict.keys()
 
    def __iter__(self):
        return self._batch
    
    def __next__(self):
        if self._pidDict.keys():
            self._createBatch()
        else :
            raise StopIteration

    def __len__():
        pass

