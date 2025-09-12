import math
import random
import numpy as np
import torch
from torch.utils.data import Sampler

class PKBatchExpander:
    def __init__(self, dataset, k = 2):
        assert k > 1 
        self._k = k
        self._dataset = dataset
        self._dict = dataset.indexMap

    def _rejectSampler(self, list, excluded , n ):
        sampleSpace = [item for item in list if item != excluded]
        if len(sampleSpace) == 0:
            return []
        elif len(sampleSpace) < n :
            return random.choices(sampleSpace, k = n)
        return random.sample(sampleSpace , k = n) 

    def sample(self, labels, indices):
        batch = []
        batchLabels = []
        batchIndexes = []
        for i , label in enumerate(labels):
            label = label.item() if torch.is_tensor(label) else label
            samples = self._dict[label]
            reject = indices[i]
            selected = self._rejectSampler(samples, reject, self._k)
            for k in selected:
                x , y , _ = self._dataset[k]
                batch.append(x)
                batchLabels.append(y)
        return torch.stack(batch) , torch.tensor(batchLabels)
            
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
        for label, indices in self._dataset.indexMap.items():
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
            if len(batch) < self._pk:
                samples = int(self._pk - len(batch))
                batch.extend(random.choices(range(len(self._dataset)), k = samples))
            batch = batch[:self._pk]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return math.ceil(self._totalSamples / self._pk)
