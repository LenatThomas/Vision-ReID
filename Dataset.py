import os
import random
import torch
import math
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Sampler
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
        self._labels = {pid : idx for idx, pid in enumerate(sorted(set(self._pids)))}
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
        pid    = splits[0]
        cid    = splits[1]
        return (pid , cid)
    
    def _buildDict(self):
        self._dict = {}
        for index , label in enumerate(self._pids):
            if label not in self._dict.keys():
                self._dict[label] = []
            self._dict[label].append(index)
    
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
            label = self._labels[pid]
            return image, label
        else :
            raise StopIteration
        
    @property
    def pids(self):
        return self._pids
        
    @property
    def labelDict(self):
        return self._dict

    @property
    def nClasses(self):
        return len(self._labels.keys())

class PKSampler(Sampler):
    def __init__(self , dataset, indices = None , p = 16 , k = 4):
        self._p = p
        self._k = k
        self._pk = p * k
        self._dataset = dataset
        self._indices = set(indices) if indices is not None else set(range(len(dataset)))
        self._reset()

    def _reset(self):
        self._labelDict = {}
        for label, indices in self._dataset.labelDict.items():
            filtered = [i for i in indices if i in self._indices]
            if filtered:
                self._labelDict[label] = filtered
        self._totalSamples = sum(len(v) for v in self._labelDict.values())
        self._labels = list(self._labelDict.keys())
        random.shuffle(self._labels)
        self._labelCounts = {label : 0 for label in self._labels}
        

    def __iter__(self):
        self._reset()
        labels = self._labels.copy()
        batch = []
        while (len(labels) != 0):
            selected = random.sample(labels, min(self._p, len(labels)))
            for l in selected:
                instances = self._labelDict[l]
                if len(instances) < self._k:
                    samples = random.choices(instances, k = self._k)
                else :
                    samples = random.sample(instances, k = self._k)
                batch.extend(samples)
                self._labelCounts[l] += self._k
                if self._labelCounts[l] >= len(instances) and l in labels:
                    labels.remove(l)
            if len(batch) == self._pk:
                random.shuffle(batch)
                yield batch
                batch = []

        if len(batch) != 0:
            print('Triggered1')
            if len(batch) < self._pk:
                print('Triggered 2')
                samples = int(self._pk - len(batch))
                batch.extend(random.choices(range(len(self._dataset)), k = samples))
            batch = batch[:self._pk]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return math.ceil(self._totalSamples / self._pk)

def mineTriplets(embeddings , labels):
    pairwise = torch.cdist(embeddings, embeddings , p = 2)
    triplets = []
    for i , label in enumerate(labels):
        posMask = (labels == label) & torch.arange(len(labels), device=labels.device) != i
        negMask = (labels != label)
        if posMask.sum() == 0 or negMask.sum() == 0:
            continue
        hardestNegative = pairwise[i][negMask].argmin()
        hardestPositive = pairwise[i][posMask].argmax()
        hardestPositive = posMask.nonzero()[hardestPositive].item()
        hardestNegative = negMask.nonzero()[hardestNegative].item()
        triplets.append((i , hardestPositive, hardestNegative))
    return triplets