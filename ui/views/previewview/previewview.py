import os
from pathlib import Path
import string
from ui.utils.components import DatasetScrollArea
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QTableView, QWidget, QVBoxLayout, QAbstractItemView, QHeaderView, QPushButton, QStackedWidget, QToolButton, QSlider, QScrollArea, QMainWindow
dir = os.path.dirname(__file__)


class PreviewView(QWidget):
    """Class for Preview Dataset Screen UI"""
    labelClicked = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.labels = {}
        self.pWindowContainer = QWidget()
        self.pWindowContainerLayout = QVBoxLayout()
        self.pWindowContainer.setWindowTitle("Preview Dataset")

        # Add Dataset Screen
        addDatasetScreen = QWidget()
        addDatasetScreen.setStyleSheet('background-color: #53433E')
        addDatasetScreenLayout = QVBoxLayout()
        self.selectDatasetButton = QPushButton("Add a folder")
        self.selectDatasetButton.setStyleSheet(
            'background-color: #A08D86; font-size:10px')
        self.selectDatasetCSVButton = QPushButton("Add a CSV")
        self.selectDatasetCSVButton.setStyleSheet(
            'background-color: #A08D86; font-size:10px')

        # define onclicks
        self.selectDatasetButton.clicked.connect(self.goToViewDataset)
        self.selectDatasetCSVButton.clicked.connect(self.goToViewDataset)

        addDatasetScreenLayout.addWidget(
            self.selectDatasetButton, QtCore.Qt.AlignTop)
        addDatasetScreenLayout.addWidget(
            self.selectDatasetCSVButton, QtCore.Qt.AlignBottom)
        addDatasetScreenLayout.addStretch()
        addDatasetScreen.setLayout(addDatasetScreenLayout)

        # View Dataset Screen
        viewDatasetWindow = QWidget()
        viewDatasetWindowLayout = QVBoxLayout()

        self.backButton = QPushButton("Back")
        self.backButton.setStyleSheet(
            'background-color: #A08D86; font-size:10px')
        self.backButton.clicked.connect(self.goToAddDataset)

        self.datasetContainer = DatasetScrollArea()

        # alphabet frequency:
        alphabetFrequency = QGridLayout()
        row = 0
        col = 0
        for letter in string.ascii_uppercase:
            label = QLabel(letter + ": 0")
            self.labels[letter] = label
            self.labels[letter].mousePressEvent = lambda event, letter=letter: self.handleLabelClick(
                letter)
            alphabetFrequency.addWidget(label, row, col)
            col += 1
            if col == 13:
                row += 1
                col = 0
        viewDatasetWindowLayout.addWidget(self.backButton)
        viewDatasetWindowLayout.addWidget(self.datasetContainer)
        viewDatasetWindowLayout.addLayout(alphabetFrequency)
        viewDatasetWindow.setLayout(viewDatasetWindowLayout)

        # stacked widget to store different screens
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(addDatasetScreen)
        self.stackedWidget.addWidget(viewDatasetWindow)

        self.pWindowContainerLayout.addWidget(self.stackedWidget)
        self.pWindowContainer.setLayout(self.pWindowContainerLayout)
        self.pWindowContainer.show()

    def goToAddDataset(self):
        """Click to go to add dataset screen"""
        self.stackedWidget.setCurrentIndex(0)

    def goToViewDataset(self):
        """Click to go to view dataset screen"""
        self.stackedWidget.setCurrentIndex(1)

    def handleLabelClick(self, letter):
        """to be documented"""
        self.labelClicked.emit(letter)
