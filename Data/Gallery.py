import torch
from tqdm import tqdm

class Gallery :
    def __init__(self, model , path, device = 'cpu'):
        self._model = model
        self._path = path
        self._device = device
        self._features = {}

    def build(self, loader):
        self._model.eval()
        with torch.no_grad():
            for images, labels, indices in tqdm(loader, desc = 'Building Dictionary'):
                images = images.to(self._device)
                _, embeddings = self._model(images)
                for i , index in enumerate(indices):
                    self._features[index] = (embeddings[i],labels[i])

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def __getitem__(self):
        pass

