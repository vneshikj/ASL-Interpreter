from collections import Counter
from PyQt5.QtWidgets import QFileDialog, QLabel, QWidget
from PyQt5.QtCore import QObject, pyqtSignal, QAbstractTableModel, Qt, QModelIndex, QVariant
from PyQt5.QtGui import QPixmap, QImage, QStandardItem, QIcon
from matplotlib.pyplot import imshow
import torchvision.datasets as datasets
from ML.modules.predict import labelToLetter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import traceback


class PreviewModel(QAbstractTableModel):
    """ Preview model. Contains functions for displaying dataset"""
    def __init__(self, directory=None):
        """Parameters
        @directory: where the dataset is located"""
        super().__init__()
        self.directory = directory
        self._data = None or []
        self._matching = self._data
        self.letter = None
        self.isFolder = False

    def rowCount(self, parent=QModelIndex()):
        """Counts the rows in dataset
        @parent: QModelIndex()"""
        return (len(self._data) + 4) // 5

    def columnCount(self, parent=QModelIndex()):
        """Counts columns in dataset
        @parent: QModelIndex()"""
        return 5

    def flags(self, index):
        """Checks for valid indices. 
        @index: int; the index to be checked
        @returns: signal; NoItemFlags (if not valid), ItemIsEnabled/ ItemIsSelectable if valid"""
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def clear(self):
        """Resets Data"""
        self.beginResetModel()
        self._data = []
        self._matching = []
        self.endResetModel()

    def data(self, index, role=Qt.DisplayRole):
        """creates image and label for our data"""
        if not index.isValid():
            return QVariant()
        row = index.row()
        self._matching = [{"image": self._data[r]["image"], "label": self._data[r]["label"]} for r in range(
            len(self._data)) if self._data[r]["label"] == self.letter] if self.letter is not None else self._data
        imageAndLabel = None
        if index.column() == 0:
            imageAndLabel = self.createImageAndLabel(row*5) 
        elif index.column() == 1:
            imageAndLabel = self.createImageAndLabel(row*5+1)
        elif index.column() == 2:
            imageAndLabel = self.createImageAndLabel(row*5+2)
        elif index.column() == 3:
            imageAndLabel = self.createImageAndLabel(row*5+3)
        elif index.column() == 4:
            imageAndLabel = self.createImageAndLabel(row*5+4)
        if imageAndLabel is None:
            return
        if role == Qt.DisplayRole:
            return imageAndLabel[1]
        elif role == Qt.DecorationRole:
            return imageAndLabel[0]
        return QVariant()

    def createImageAndLabel(self, row):
        """Creates image and its label from CSV file
        @row: int row where the image is located in image data
        @return: pixmap of image and image label"""
        if row > len(self._matching)-1:
            return
        image = Image.fromarray(self._matching[row]["image"])
        # Convert image to grayscale if it's not already in grayscale format
        if image.mode != 'L':
            image = image.convert('L')
        qimage = QImage(
            image.tobytes(), image.width, image.height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        scaledPixMap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)
        label = str(self._matching[row]["label"])
        return (scaledPixMap, label)
  
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """to be documented"""
        if role != Qt.DisplayRole:
            return QVariant()

        if orientation == Qt.Horizontal:
            if section == 0:
                return "Label"
            elif section == 1:
                return "Image"

        return QVariant()

    def addData(self, data):
        """appends data to the end of existing data (images) for folder conversion
        @data: data to be appended"""
        self.beginInsertRows(
            QModelIndex(), self.rowCount() * 5-4, self.rowCount() * 5-4)
        self._data.append(data)
        self.endInsertRows()

    def updateDirectory(self):
        """Updates the directory of the dataset"""
        try:
            options = QFileDialog.Options()
            filePath = QFileDialog.getExistingDirectory(
                None, "Select Folder with DataSet")
            if not filePath or filePath == "":
                return
            self.directory = str(filePath)
            self.finished.emit()
            print("directory:", self.directory)
        except:
            traceback.print_exc()
            self.directory = None
            return

    def updateDirectoryCSV(self):
        """Updates directory of the dataset"""
        try:
            self.isFolder = False
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(
                None, "Select Folder or .CSV", "", "All files (*.*)", options=options)
            if not filePath or str(filePath)[-4:] != ".csv":
                return
            self.directory = str(filePath)
            print("directory", self.directory)
        except:
            traceback.print_exc()
            self.directory = None
            return

    def getDirectory(self):
        """Gets the directory of the dataset
        @Return: String; path to dataset"""
        return self.directory

    def convertCSVToNumpy(self):
        """Converts CSV to Numpy array and resizes to 28x28"""
        try:
            print("Reading CSV...")
            self.clear()
            df = pd.read_csv(self.directory, skiprows=[0])
            labels = df.iloc[:, 0].values
            imArray = df.iloc[:, 1:].values
            imArray = imArray.astype(np.uint8)  # Normalize as uint8

            # Resize each image to (28, 28) using OpenCV
            resizedImages = []
            for image in imArray:
                resizedImage = cv2.resize(image.reshape(
                    (28, 28)), (28, 28), interpolation=cv2.INTER_LINEAR)
                resizedImages.append(resizedImage)

            # Create a list of dictionaries for each image
            for idx, image in enumerate(resizedImages):
                self.addData(
                    {"label": labelToLetter(labels[idx]), "image": image})
            self.countLabels()
            print(len(self._data))
            print("Converted to NumPy!")
        except:
            traceback.print_exc()
            return

    def convertFolderToNumpy(self):
        """Converts a folder of images to numpy array. adds image data to self.data"""
        try:
            self.clear()
            self.isFolder = True
            dataset = datasets.ImageFolder(
                root=self.directory,
                transform=transforms.Compose([transforms.Grayscale()]),
            )
            # Get the classes
            classes = dataset.classes
            # Create a list of dictionaries
            for i in range(len(dataset)):
                img, labelIndex = dataset[i]
                label = classes[labelIndex]
                imgArray = np.array(img)
                dataDict = {'image': imgArray, 'label': label.upper()}
                self.addData(dataDict)
            self.countLabels()
            self.finished.emit()
        except Exception:
            traceback.print_exc()
            self.clear()
            return
    
    def countLabels(self):
        labels = [d["label"] for d in self._data]
        labelFreq = Counter(labels)
        self.frequencyUpdated.emit(labelFreq)
        self.finished.emit()


    def filter(self, letter):
        """Filters out images and labels to only show inputted letter
        @letter: String; letter to be shown"""
        if self.letter == letter:
            self.letter = None
            return
        self.letter = letter

    frequencyUpdated = pyqtSignal(Counter)
    finished = pyqtSignal()
