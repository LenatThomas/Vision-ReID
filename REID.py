from scipy.io import loadmat
import os

# Dont forget to apply transformation on the bounding boxes, okay!!

class BoundingBoxes():
    def __init__(self, mat):
        self._fileName = mat[0][0]
        self._nPersons = mat[1][0][0]
        boxes = mat[2][0]
        self._boxes = []
        self._boxdims = []
        for b in boxes :
            x, y, w, h = map(int, b['idlocate'][0])
            self._boxdims.append((w, h))
            x0 = x + w
            y0 = y + h
            box = [x , y , x0 , y0]
            self._boxes.append(box)

    @property
    def dimensions(self):
        return self._boxdims
    
    @property
    def file(self):
        return self._fileName
    
    @property
    def nPersons(self):
        return self._nPersons
    
    @property
    def boxes(self):
        return self._boxes
    
    def __len__(self):
        return self._nPersons
    
    def __getitem__(self, index):
        if index < 0 or index > self._nPersons:
            raise IndexError('Index out of range')
        return self._boxes[index]
    



class PersonLoader():
    def __init__(self, path , queryPath):
        self._path = path
        self._queryPath = queryPath
        self._data = loadmat(path)['Img'][0]
        self._persons = loadmat(queryPath)['Person'][0]
        self._len  = len(self._data)
        self._constructBoxes()

    def _constructBoxes(self):
        self._boxes = {}
        for b in range(self._len):
            mat = self._data[b]
            box = BoundingBoxes(mat = mat)
            file = box.file
            self._boxes[file] = box

    def __getitem__(self, index):
        if index < 0 or index > self._len:
            raise IndexError('Index out of range')
        return self._boxes[index]
    
    def __len__(self):
        return self._len
    
loader = PersonLoader(path = 'data/annotation/Images.mat' , queryPath = 'data/annotation/Person.mat')
mat = loadmat('data/annotation/Person.mat')['Person'][0]
print(mat[0][0][0])
print(mat[0][1][0][0])
print(mat[0][2][0])