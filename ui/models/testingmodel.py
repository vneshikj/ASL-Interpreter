from PyQt5.QtCore import QObject, pyqtSignal, QThread, QRunnable, QThreadPool
from ML.modules.predict import predictImage, mnistToNpArray, resizeImageTo28x28Greyscale, resizeMnistTo28x28Greyscale
from ML.modules.models import LeNet, AlexNet, VGG
import numpy as np
from torchvision.transforms import transforms
from PIL import ImageQt
from utils.constants import NUM_CHANNELS, NUM_CLASSES


class PredictionRunnable(QRunnable):
    def __init__(self, testingModel):
        super().__init__()
        self.testingModel = testingModel

    def run(self):
        if self.testingModel.modelToUse is None:
            return
        self.testingModel.loading.emit(True)
        self.testingModel.currentIndex = None
        for image in self.testingModel.testingImages:
            self.testingModel.predictionResults.append(
                self.testingModel.makePrediction(image))
        if self.testingModel.currentIndex is None and len(self.testingModel.predictionResults) > 0:
            self.testingModel.currentIndex = 0
        self.testingModel.loading.emit(False)
        self.testingModel.notifyIndexChanged()


class LoadCSVRunnable(QRunnable):
    def __init__(self, testingModel, csvPath):
        super().__init__()
        self.testingModel = testingModel
        self.csvPath = csvPath

    def run(self):
        self.testingModel.loading.emit(True)
        self.testingModel.testingImages = mnistToNpArray(self.csvPath)
        self.testingModel.loading.emit(False)


class TestingModel(QObject):
    """The model for the view that is responsible for predictions, contains functions relating to using a model to predict as well as loading images for predicting"""
    predictionsUpdated = pyqtSignal(dict)
    loading = pyqtSignal(bool)

    def __init__(self, loadingModel):
        super().__init__()
        self.modelManager = loadingModel
        self.testingImages = []
        self.predictionResults = []
        self.modelToUse = None
        self.currentIndex = None
        self.threadPool = QThreadPool()
        self.isLoading = False

    def updateModel(self, model):
        """Parameters:
            @model: the model in the form of a pickled dictionary containing all of the model information (the dictionary that is produced when a trained model is saved)
            """
        self.modelToUse = model
        modelName = self.modelToUse["metadata"]["architecture"]
        if modelName == "LeNet":
            model = LeNet(NUM_CHANNELS, NUM_CLASSES)
        elif modelName == "AlexNet":
            model = AlexNet(NUM_CHANNELS, NUM_CLASSES)
        elif modelName == "VGG":
            model = VGG(NUM_CHANNELS, NUM_CLASSES)
        model.load_state_dict(self.modelToUse["model_state_dict"])
        self.modelToUse = model

    def loadImageGroup(self, filePaths):
        """Parameters:
            @filePaths: a list of filepaths for images in standards format
        This function sets up a group of images for prediction
        """
        self.testingImages.clear()
        self.predictionResults.clear()
        for filePath in filePaths:
            self.testingImages.append(filePath)
        self.startPredictions()

    def loadCSV(self, csvPath):
        """Parameters:
            @csvPath: the filepath to a MNIST dataset 
        This function setus up a group of images from predction from a MNIST dataset format
        """
        self.testingImages.clear()
        self.predictionResults.clear()
        runnable = LoadCSVRunnable(self, csvPath)
        self.threadPool.start(runnable)

    def goNext(self):
        """Updates the current prediction to display the next prediction from the overall list of predictions"""
        if self.currentIndex is None or self.currentIndex == len(self.predictionResults) - 1:
            return
        self.currentIndex += 1
        self.notifyIndexChanged()

    def goBack(self):
        """Updates the current prediction to display the previous prediction from the overall list of predictions"""
        if self.currentIndex is None or self.currentIndex == 0:
            return
        self.currentIndex -= 1
        self.notifyIndexChanged()

    def notifyIndexChanged(self):
        self.predictionsUpdated.emit(self.predictionResults[self.currentIndex])

    def loadImage(self):
        raise Exception("not implemented")

    def startPredictions(self):
        """Makes predictions on the list of prediction images based on the model that was provided"""
        if self.modelToUse is None:
            return

        # create a new runnable and move it to the thread pool
        runnable = PredictionRunnable(self)
        self.threadPool.start(runnable)

    def makePrediction(self, image):
        """Prediction function for individual image"""
        if self.modelToUse is None or self.isLoading:
            return
        prediction = {}
        if not isinstance(image, np.ndarray):
            prediction["image"] = image
            pred = predictImage(resizeImageTo28x28Greyscale(image), self.modelToUse)
            prediction["prediction"] = pred[0]
            prediction["accuracy"] = pred[1]
        else:
            prediction["image"] = image
            pred = predictImage(resizeMnistTo28x28Greyscale(image), self.modelToUse)
            prediction["prediction"] = pred[0]
            prediction["accuracy"] = pred[1]
        return prediction

    def predictOnCapture(self, capturedImage):
        """Makes a prediction based on the last image that the user captured on the webcam feed
        @capturedImage: image last captured from webcam"""
        if self.modelToUse is not None:

            # Convert the QImage to a PIL image
            pilImage = ImageQt.fromqimage(capturedImage)

            # Resize and convert the image to a 28x28 RGB image
            imageTransform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                # normalize the image
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            transformedImage = imageTransform(pilImage)
            image = np.array(transformedImage)
            # change the order of dimensions
            image = np.transpose(image, (1, 2, 0))
            prediction = predictImage(image, self.modelToUse)
            return prediction
