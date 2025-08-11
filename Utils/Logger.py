import logging
from logging import Formatter, FileHandler, StreamHandler

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