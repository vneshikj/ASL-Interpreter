from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QRunnable, QThreadPool
import numpy as np
from ML.modules.train import Train
from ML.modules.models import LeNet, AlexNet, VGG
from utils.constants import NUM_CHANNELS, NUM_CLASSES


def tensorToNumpy(tensor):
    """Converts a tensor into a numpy array for conversion.
    @Returns: Array; numpy array"""
    # unnormalize tensor
    tensor = tensor * 0.5 + 0.5
    # convert tensor to numpy array
    np_array = tensor.numpy()
    # move channel dimension to last axis
    np_array = np_array.transpose((1, 2, 0))
    # convert to integer 0-255
    np_array = np_array * 255
    # convert to uint8
    np_array = np_array.astype('uint8')
    return np_array


class LoadThread(QRunnable):
    """
    Thread to load the datasets on to avoid blocking the main thread
    """
    def __init__(self, train, directory, isMnist=True):
        """Parameters
        @train: a reference to the train class that contains all the needed information for training
        """
        super().__init__()
        self.train = train
        self.directory = directory
        self.isMnist = isMnist

    def run(self):
        if self.isMnist:
            self.train.loadMnistDataset(self.directory)
        else:
            self.train.loadFolderDataset(self.directory)


class TrainRunnable(QRunnable):
    def __init__(self, train):
        super().__init__()
        self.train = train

    def run(self):
        self.train.startTraining()


class TrainModel(QObject):
    """Train Model class. This class communicates with the controller and calls functions from
    train class."""
    directoryUpdated = pyqtSignal()
    percentageChanged = pyqtSignal(float)
    logsChanged = pyqtSignal(str)
    trainingStarted = pyqtSignal()
    trainingCancelled = pyqtSignal()
    trainingComplete = pyqtSignal(bool)
    remainingTimeChanged = pyqtSignal(float)
    datasetLoaded = pyqtSignal(bool)
    splitDone = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.directory = None
        self.train = Train()
        self.presetModels = ["LeNet", "AlexNet", "VGG"]
        self.train.percentageChanged.connect(self.onPercentageChanged)
        self.train.logsChanged.connect(self.onLogsChanged)
        self.train.trainingStarted.connect(self.onTrainingStarted)
        self.train.trainingCancelled.connect(self.onTrainingCancelled)
        self.train.trainingComplete.connect(self.onTrainingComplete)
        self.train.remainingTimeChanged.connect(self.updateRemainingTime)
        self.train.datasetLoaded.connect(self.onDatasetLoaded)
        self.train.splitDone.connect(self.updateTrainValRatio)
        self.isTraining = False

    def updateDirectory(self):
        """Updates the directory where the dataset is located. Saves to self.directory."""
        options = QFileDialog.Options()
        filePath = QFileDialog.getExistingDirectory(
            None, "Select Folder with DataSet")
        if not filePath or filePath == "":
            return
        self.directory = str(filePath)
        self.directoryUpdated.emit()
        runnable = LoadThread(self.train, self.directory, isMnist=False)
        QThreadPool.globalInstance().start(runnable)

    def updateDirectoryCSV(self):
        """Updates the directory where the CSV dataset is located. Saves to self.directory."""
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            None, "Select Folder or .CSV", "", "All files (*.*)", options=options)
        if not filePath or str(filePath)[-4:] != ".csv":
            return
        self.directory = str(filePath)
        self.directoryUpdated.emit()
        runnable = LoadThread(self.train, self.directory, isMnist=True)
        QThreadPool.globalInstance().start(runnable)

    def openDirectory(self):
        """Opens the directory where the dataset is located."""
        print(self.directory)


    def updateHyperparameters(self, hyperparameterName, value):
        """Updates hyperparameters
        @hyperparameterName: String; name of hyperparameters to be updated.
        @value: int; new value of hyperparameter."""
        self.train.setHyperparameters(hyperparameterName, value)

    def onLogsChanged(self, log):
        """Updates the log value when a log is changed."""
        self.logsChanged.emit(log)

    def onPercentageChanged(self, percentage):
        """Updates the percentage value when the percentage is changed."""
        self.percentageChanged.emit(percentage)

    def getPresetModels(self):
        """Gets our preset models for model select dropdown."""
        return self.presetModels

    def updateTrainRatio(self, ratio):
        """Updates train ratio for validation and training data
        @ratio: int; the percentage of training vs validation data in training dataset."""
        self.train.trainRatio = ratio/100
        self.train.trainTestSplit()
    
    def updateTrainValRatio(self, trainVal):
        """Updates the training/ validation ratio
        @trainVal: int new training/ validation ratio"""
        self.splitDone.emit(trainVal)

    def usePresetModel(self, name):
        """Chooses a preset model and calls the model
        @name: String; the name of the model"""
        model = None
        if name == "LeNet":
            model = LeNet(numChannels=NUM_CHANNELS, classes=NUM_CLASSES)
        elif name == "AlexNet":
            model = AlexNet(numChannels=NUM_CHANNELS, classes=NUM_CLASSES)
        elif name == "VGG":
            model = VGG(numChannels=NUM_CHANNELS, classes=NUM_CLASSES)
        print(name)
        self.train.initModel(model, name)
        
    def getDirectory(self):
        """Gets the name of the directory of the dataset. Calls from train class.
        @Return: the name of the directory of the dataset"""
        return self.directory

    def saveModel(self, directory):
        """"Saves the trained model in inputted directory. Calls from train class."""
        self.train.saveModel(directory)

    def startTraining(self):
        """Starts the training process. Sets isTraining to true and gives it its own thread.
        """
        self.train.percentageChanged.connect(self.percentageChanged.emit)
        if self.train.savedModelParameters is not None:
            name = self.train.savedModelParameters["architecture"]
            if name == "LeNet":
                self.usePresetModel("LeNet")
            elif name == "AlexNet":
                self.usePresetModel("AlexNet")
            elif name == "VGG":
                self.usePresetModel("VGG")
        runnable = TrainRunnable(self.train)
        QThreadPool.globalInstance().start(runnable)
        self.isTraining = self.train.isTraining

    def stopTraining(self):
        """Stops the training process. Called from train class."""
        self.train.stopTraining()
        self.isTraining = self.train.isTraining

    def onTrainingCancelled(self):
        """Tells controller that the training process is cancelled."""
        self.isTraining = False
        self.trainingCancelled.emit()

    def onTrainingStarted(self):
        """Tells controller that the training process has started."""
        self.isTraining = True
        self.trainingStarted.emit()

    def onTrainingComplete(self, isComplete):
        """Tells controller that the training process has completed.
        @isComplete: boolean; signal."""
        self.trainingComplete.emit(isComplete)
        if isComplete:
            self.isTraining = False
            self.trainingCancelled.emit()

    def onDatasetLoaded(self, isLoaded):
        """Tells controller that dataset is loaded
        @isLoaded: boolean; signal"""
        self.datasetLoaded.emit(isLoaded)

    def updateRemainingTime(self, time):
        """Tells controller to update the time remaining
        @time: boolean; signal"""
        self.remainingTimeChanged.emit(time)
