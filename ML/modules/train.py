from PyQt5.QtCore import QObject, pyqtSignal
import time
from torch import nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from ML.modules.ConvertDataset import ConvertDataset
import torch
import matplotlib
matplotlib.use("Agg")

# our neural network
# used to display a classification report on our testing set
# constructs random training and testing split from input set of data
# optimizer

# HYPERPARAMETERS
# ------------ TO BE INITIALISED WITH UI --------------
INIT_LR = 1e-3  # learning rate
BATCH_SIZE = 64
EPOCHS = 5

# define the train and val splits- CAN BE CHANGED
# ------------ TO BE INITIALISED WITH UI --------------
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1-TRAIN_SPLIT


class Train(QObject):
    """Train Class. Contains functions for training including selection 
    of dataset, model, hyper parameters and training process. Functions are called from Train Model
    class.
    """
    percentageChanged = pyqtSignal(float)
    logsChanged = pyqtSignal(str)
    trainingCancelled = pyqtSignal()
    trainingStarted = pyqtSignal()
    trainingComplete = pyqtSignal(bool)
    remainingTimeChanged = pyqtSignal(float)
    datasetLoaded = pyqtSignal(bool)
    splitDone = pyqtSignal(tuple)
    def __init__(self):
        super().__init__()
        self.isTraining = False
        self.dataset = None
        self.model = None
        self.opt = None
        self.hyperParameters = {"BatchSize": 64, "Epochs": 5}
        self.savedModelParameters = None
        self.trainRatio = 0.5
        self.trainSet = None
        self.validSet = None
        self.modelArchitecture = ""
        self.logs = ""
        self.trainPercentage = 0
        # set device used to trian model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    # ------------- INITIALIZE DICTIONARY TO STORE TRAINING HISTORY -------------
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

    def getDataset(self):
        """Gets the dataset
            @Return: the dataset"""
        return self.dataset

    def getPercentage(self):
        """Gets percentage of training process
        @Return: int; percentage of training completed"""
        return self.trainPercentage

    def getLogs(self):
        """Gets logs
        @Return: String; logs"""
        return self.logs

    def setHyperparameters(self, hyperparameterName, value):
        """Sets HyperParameters for training.
        @hyperparameterName: String; the name of the hyperparameter
        @value: int; new value of inputted hyperparameter"""
        if self.isTraining:
            return
        if hyperparameterName in self.hyperParameters:
            self.hyperParameters[hyperparameterName] = value
        else:
            raise Exception("Non-Existent hyperparameter")

    def initModel(self, model, architecture):
        """Initialises the model
        @model: model; the selected model
        @architecture: String; the architecture of the selected model """
        if self.isTraining:
            return
        # ------------- INITIALIZE MODEL -------------
        print("[INFO] initializing the model...")
        self.modelArchitecture = architecture
        self.model = model

    def loadMnistDataset(self, csvPath):
        """Loads MNIST dataset by converting it into a dataloader
        @csvPath: String; the path where the training CSV is located"""
        if self.isTraining:
            return
        # ------------- MNIST IMPLEMENTATION --------------
        print("[INFO] loading dataset...")
        self.datasetLoaded.emit(False)

        trainData = ConvertDataset(csvPath)
        dataLoader = DataLoader(trainData, shuffle=True,
                                batch_size=self.hyperParameters["BatchSize"])
        self.dataset = dataLoader
        self.trainTestSplit()
        self.datasetLoaded.emit(True)

    def loadFolderDataset(self, folderPath):
        try:
            if self.isTraining:
                return
            self.datasetLoaded.emit(False)
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            dataset = ImageFolder(folderPath, transform=transform)
            dataloader = DataLoader(
                dataset, batch_size=self.hyperParameters["BatchSize"], shuffle=True)
            self.dataset = dataloader
            self.trainTestSplit()
            self.datasetLoaded.emit(True)
        except FileNotFoundError:
            print("Not in correct format!")
            return

    def trainTestSplit(self):
        """Handles the split between in the training data between training and validation, 
            converts each into a dataloader uses hyperparameters to key in batch size and split
            dataset.

        """
        # ------------- TEST/ VALIDATION SPLIT -------------
        # randomly split training data for validation data
        # Define the indices for the training and validation sets
        if self.dataset is None or self.isTraining:
            return

        train_size = int(len(self.dataset.dataset) * self.trainRatio)
        test_size = len(self.dataset.dataset) - train_size
        train_set, test_set = random_split(
            self.dataset.dataset, [train_size, test_size])

        trainLoader = DataLoader(
            train_set, batch_size=self.hyperParameters["BatchSize"], shuffle=True)
        valLoader = DataLoader(
            test_set, batch_size=self.hyperParameters["BatchSize"], shuffle=True)
        self.trainSet = trainLoader
        self.validSet = valLoader

        self.trainSteps = len(self.trainSet.dataset)  # batch size
        self.valSteps = len(self.validSet.dataset)  # batch size

        self.splitDone.emit((len(self.trainSet.dataset), len(self.validSet.dataset)))

    def saveModel(self, directory):
        """Saves the trained model. 
        @directory: the path where you want to save the model."""
        if self.opt is None or self.savedModelParameters is None:
            return
        metadata = self.savedModelParameters
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "metadata": metadata
                    }, directory)

    def stopTraining(self):
        """Stops training process"""
        self.isTraining = False
        self.logs += "Training Stopped"
        self.logsChanged.emit(self.logs)

    def startTraining(self):
        """Starts training process. Checks for a loaded dataset and chosen model"""
        if self.trainSet is None or self.validSet is None or self.isTraining:
            self.logs = "No dataset Loaded! Go to the \"Dataset\" tab to load a dataset."
            self.logsChanged.emit(self.logs)
            self.trainingCancelled.emit()
            return
        self.trainingComplete.emit(False)
        self.savedModelParameters = {"architecture": self.modelArchitecture,
                                     "num_epochs": self.hyperParameters["Epochs"],
                                     "batch_size": self.hyperParameters["BatchSize"],
                                     "train_ratio": self.trainRatio}
        # ------------- INITIALIZE OPTIMIZER AND LOSS FUNCTION -------------
        lossFn = nn.NLLLoss()
        # ------------- MEASURE HOW LONG TRAINING WILL TAKE -------------
        print("[INFO] training the network...")
        startTime = time.time()
        self.opt = Adam(self.model.parameters(), lr=INIT_LR)
        self.isTraining = True
        self.logs = "Training Started...\n"
        self.trainPercentage = 0
        self.trainingStarted.emit()
        self.logsChanged.emit(self.logs)
        self.percentageChanged.emit(self.trainPercentage)
        # ------------- TRAINING LOOP -------------
        for e in range(0, self.hyperParameters["Epochs"]):
            self.model.train()  # put model in train mode
            elapsedTime = time.time() - startTime
            remainingTime = elapsedTime * \
                (self.hyperParameters["Epochs"] - e - 1) / (e + 1)
            self.remainingTimeChanged.emit(remainingTime)
            # initialize training and validation loss for current epoch
            totalTrainLoss = 0
            totalValLoss = 0

            # initialize the number of correct predictions in the training and validation step
            trainCorrect = 0
            valCorrect = 0

            for (x, y) in self.trainSet:
                if not self.isTraining:
                    self.trainingCancelled.emit()
                    return
                # send input to device
                (x, y) = (x.to(self.device), y.to(self.device))

                # perform forward pass and calculate training loss
                pred = self.model(x)
                loss = lossFn(pred, y)

                # zero out gradients, perform backpropogation and update weights
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # add loss to the total training loss so far and calculate num of correct predictions
                # book keeping variables
                totalTrainLoss += loss
                trainCorrect += (torch.argmax(torch.softmax(pred, dim=1), 1) == y).type(torch.float).sum().item()

            # switch off autograd for evaluation
            with torch.no_grad():
                # set model in evaluation mode
                self.model.eval()
                # loop over validation set
                for (x, y) in self.validSet:
                    # send input to device
                    (x, y) = (x.to(self.device), y.to(self.device))

                    # make predictions and calculate validation loss
                    pred = self.model(x)
                    totalValLoss += lossFn(pred, y)

                    # calculate number of correct predictions
                    valCorrect += (torch.argmax(torch.softmax(pred, dim=1), 1) == y).type(torch.float).sum().item()

            # -------- CALCULATE AVERAGE TRAINING AND VALIDATION LOSS ----------
            avgTrainLoss = totalTrainLoss / self.trainSteps
            avgValLoss = totalValLoss / self.valSteps

            # -------- CALCULATE TRAINING AND VALID ACCURACY ----------
            trainCorrect = trainCorrect / len(self.trainSet.dataset)
            valCorrect = valCorrect / len(self.validSet.dataset)

            # -------- UPDATE TRAINING HISTORY -----------
            self.history["train_loss"].append(
                avgTrainLoss.cpu().detach().numpy())
            self.history["train_accuracy"].append(trainCorrect)
            self.history["val_loss"].append(avgValLoss.cpu().detach().numpy())
            self.history["val_accuracy"].append(valCorrect)

            # print the model training and validation information
            self.logs += "[INFO] EPOCH: {}/{}".format(
                e+1, self.hyperParameters["Epochs"]) + "\n"
            self.logs += "Train loss: {:.6f}, Train Accuracy: {:.4f}".format(
                avgTrainLoss, trainCorrect) + "\n"
            self.logs += "Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
                avgValLoss, valCorrect) + "\n"
            self.logsChanged.emit(self.logs)
            self.trainPercentage = 100 * \
                ((e+1)/(self.hyperParameters["Epochs"]+1))
            self.percentageChanged.emit(self.trainPercentage)

        # finish measuring how long training took
        endTime = time.time()
        self.logs += "[INFO] total time taken to train the model: {:.2f}s".format(
            endTime-startTime)
        self.trainPercentage = 100
        self.percentageChanged.emit(self.trainPercentage)
        self.logsChanged.emit(self.logs)
        self.isTraining = False
        self.trainingComplete.emit(True)
