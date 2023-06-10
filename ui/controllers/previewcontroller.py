from threading import Thread
from PyQt5.QtWidgets import QGridLayout, QStyledItemDelegate, QApplication, QWidget, QVBoxLayout, QLabel, QStyle
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QRunnable, QThreadPool, Qt, QRect, QSize, QByteArray
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from ML.modules.train import Train
from ML.modules.models import LeNet, AlexNet
from ui.utils.runnables import ConvertCsvTask
from ui.models.previewmodel import PreviewModel


class PreviewController(QObject):
    """Preview Controller class. Communicates with the view and its model. """
    def __init__(self, view, model, directory=None):
        """Parameters
        @view: view for preview window
        @model: the model for preview window"""
        super().__init__()
        self.model = model
        self.view = view        
        #self.model.finished.connect(self.onCsvConversionFinished)
        self.model.frequencyUpdated.connect(self.updateFrequency)
        self.view.labelClicked.connect(self.filterLetter)
        self.view.datasetContainer.setTableModel(self.model)
        self.view.selectDatasetButton.clicked.connect(self.updateDirectory)
        self.view.selectDatasetCSVButton.clicked.connect(
            self.updateDirectoryCSV)
        self.directory = directory
        if self.directory is not None:
            if self.directory[-4:] == ".csv":
                self.model.directory = self.directory
                self.view.goToViewDataset()
                self.model.convertCSVToNumpy()


    def updateDirectory(self):
        """Updates directory of dataset and takes user to the preview window"""
        self.model.updateDirectory()
        if self.model.directory is not None:
            self.view.goToViewDataset()
            task = ConvertCsvTask(self.model, False)
            # Submit the task to a thread pool to run on another thread
            threadPool = QThreadPool.globalInstance()
            threadPool.start(task)

    def updateDirectoryCSV(self):
        """Updates the directory where the CSV dataset is located"""
        self.model.updateDirectoryCSV()
        # print("directory:", self.model.directory)
        if self.model.directory is not None:
            self.view.goToViewDataset()
            # Create a task to convert the CSV to NumPy array
            task = ConvertCsvTask(self.model, True)
            # Submit the task to a thread pool to run on another thread
            threadPool = QThreadPool.globalInstance()
            threadPool.start(task)

    def updateFrequency(self, labelFreq):
        """Updates the labels' frequency counts
        @labelFreq: contains labels and their frequency"""
        for label, freq in labelFreq.items():
            self.view.labels[label].setText(f'{label}: {freq}')

    def filterLetter(self, letter):
        """Filters the letters. Only shows inputted letter
        @letter: the letter you want to remain visible"""
        if letter == self.model.letter:
            self.view.labels[letter].setStyleSheet("")
        else:
            if self.model.letter is not None:
                self.view.labels[self.model.letter].setStyleSheet("") 
            self.view.labels[letter].setStyleSheet("background-color:#FFB599;")
        self.model.filter(letter)
