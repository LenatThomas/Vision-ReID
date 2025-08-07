import logging
from logging import Formatter, StreamHandler, FileHandler

class TrainingTracker():
    def __init__(self):
        self._resetState()

    def _resetState(self):
        self._loss    = 0.0
        self._correct = 0
        self._total   = 0

    def reset(self):
        self._resetState()

    def update(self, loss , batch , predicted , labels):
        self._loss  += loss.item() * batch
        self._total += batch
        self._correct += (predicted == labels).sum().item()

    @property
    def accuracy(self):
        return self._correct / max(1, self._total)
    
    @property
    def loss(self):
        return self._loss / max(1, self._total)
    
    def __str__(self):
        return f"Loss: {self.loss:.4f} | Acc: {self.accuracy:.4f}"

def setupLogger(logFile='training.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = Formatter(
        fmt='%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fileHandler = FileHandler(logFile, mode='w')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    consoleHandler = StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger