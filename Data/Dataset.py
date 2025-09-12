import os
import numpy as np
import torchvision.transforms as T
from PIL import Image
from Data.Templates import ReidentificationDataset, SearchDataset
from scipy.io import loadmat

class Market1501IdentificationSet(ReidentificationDataset):
    def __init__(self, root, split = "bounding_box_train",transform = None):
        super().__init__(root, transform or T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        self._imageDir = os.path.join(self._root, split)
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
    
class CuhkSysuIdentificationSet(ReidentificationDataset):
    def __init__(self, root , split = 'Train.mat', transform = None):
        super().__init__(root, transform or T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        self._annotationDir = os.path.join(root, 'annotation', 'test' , 'train_test', split)
        self._imageDir = os.path.join(root, 'Image', 'SSM')
        self._raw = loadmat(self._annotationDir, squeeze_me=True, struct_as_record=False)
        self._mat = self._raw[list(self._raw.keys())[-1]]
        self._length = 0
        self._crops = []
        self._build()

    def _unpack(self, index):
        row     = self._mat[index]
        pid     = row.idname
        scene   = []
        for s in row.scene:
            imname   = s.imname
            box     = np.array(s.idlocate, dtype = np.int32)
            scene.append((imname, pid, box))
        return scene

    def _buildDict(self):
        self._dict = {}
        for index, pid in enumerate(self._pids):
            if pid not in self._dict.keys():
                self._dict[pid] = []
            self._dict[pid].append(index)

    def _build(self):
        self._crops = []
        for i in range(len(self._mat)):
            self._crops.extend(self._unpack(i))
        self._length = len(self._crops)
        self._pids = [i[1] for i in self._crops]
        self._buildDict()
         
    def __getitem__(self, index):
        if not (0 <= index < self._length):
            raise IndexError(f"Index {index} out of range of length {self._length}")
        imname = self._crops[index][0]
        pid    = self._crops[index][1]
        imagePath = os.path.join(self._imageDir, imname)
        image = Image.open(imagePath).convert('RGB')
        x, y, w, h    = self._crops[index][2]
        width, height = image.size
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box for {imname}: "
                f"(x={x}, y={y}, w={w}, h={h}, img_size=({width},{height}))"
            )
        crop = image.crop((x1, y1 , x2 , y2))
        crop = self._transform(crop)
        return crop, pid, index

    def filename(self, index):
        if not (0 <= index < self._length):
            raise IndexError('Index out of range')
        imname = self._crops[index][0]
        pid = self._crops[index][1]
        return f'{imname}_{pid}'

class CuhkSysuSearchSet(SearchDataset):
    def __init__(self, root, split = 'Train.mat' ,transform = None):
        super().__init__(root, transform)
        self._annotationDir = os.path.join(root, 'annotation', 'test' , 'train_test', split)
        self._imageDir = os.path.join(root, 'Image', 'SSM')
        self._raw = loadmat(self._annotationDir, squeeze_me=True, struct_as_record=False)
        self._mat = self._raw[list(self._raw.keys())[-1]]
        self._crops = []
        self._buildMapping()

    def _unpack(self, index):
        row     = self._mat[index]
        scene   = []
        for s in row.scene:
            imname   = s.imname
            box     = np.array(s.idlocate, dtype = np.int32)
            scene.append((imname, box))
        return scene

    def _buildMapping(self):
        self._crops = []
        for i in range(len(self._mat)):
            self._crops.extend(self._unpack(i))
        self._length = len(self._crops)
        self._nImages = len(set(c[0] for c in self._crops))
        self._cropMaps = self._crops

    def __getitem__(self, index):
        imname = self._crops[index][0]
        box    = self._crops[index][1]
        imagePath = os.path.join(self._imageDir, imname)
        image = Image.open(imagePath).convert('RGB')
        x, y, w, h    = self._crops[index][1]
        width, height = image.size
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x+w), min(height, y+h)
        crop = image.crop((x1, y1 , x2 , y2))
        crop = self._transform(crop)
        return crop, imname, box

        

