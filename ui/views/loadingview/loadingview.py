import os
from pathlib import Path
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QListView, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QSizePolicy, QSlider, QGridLayout

dir = os.path.dirname(__file__)


class LoadingView(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setStyleSheet(
            Path(os.path.join(dir, 'loadingview.qss')).read_text())

        # Layout is a window (QVBoxLayout) with a title container and a tabs container which is a QHBoxlayout with Load Model on the left
        # and model information on the right

        # Create title widget
        titleContainer = QWidget()
        titleLayout = QHBoxLayout()
        title = QLabel("Load Model")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        # Create back button
        backButton = QToolButton()
        backButton.setIcon(QtGui.QIcon(
            "ui/styles/theme/primary/leftarrow.svg"))
        backButton.clicked.connect(self.controller.showLandingView)
        titleLayout.addWidget(backButton, QtCore.Qt.AlignLeft)
        titleLayout.addWidget(title, QtCore.Qt.AlignRight)
        titleContainer.setLayout(titleLayout)

        # Create uploadsTab widget
        uploadsTab = QWidget()
        uploadsTab.setStyleSheet(
            "background-color: #53433E; border-radius: 4px;")

        # Create layout for uploads container
        uploadsTabLayout = QVBoxLayout()
        uploadModelsTitle = QLabel("Uploaded Models")
        uploadModelsTitle.setStyleSheet("font-size: 24px; font-weight: bold;")
        uploadsTabLayout.addWidget(uploadModelsTitle, 0, QtCore.Qt.AlignTop)
        self.uploadedModels = QListView()
        uploadsTabLayout.addWidget(self.uploadedModels)
        self.loadModelButton = QPushButton(
            "Add new model", self)  # add button to upload model

        uploadsTabLayout.addWidget(self.loadModelButton, QtCore.Qt.AlignBottom)
        uploadsTab.setLayout(uploadsTabLayout)
        uploadsTab.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.MinimumExpanding
        )

        # Right Container is for looking at model information
        modelInfoTab = QWidget()
        modelInfoTab.setStyleSheet(
            "background-color: #53433E; border-radius: 4px;")

       # Add widgets to modelInfoTab
        modelInfoTabLayout = QVBoxLayout()
        modelInfoTitle = QLabel("Model Information")
        modelInfoTitle.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.modelFileName = QLabel("Filename: ")
        self.modelFileSize = QLabel("FileSize: ")
        self.modelEpoch = QLabel("Epoch: ")
        self.modelBatchSize = QLabel("Batch Size: ")
        self.modelTrainValRatio = QLabel("Train Ratio: ")
        self.modelType = QLabel("Model Type: ")
        self.modelDelete = QPushButton("Delete Model", self)
        modelInfoTabLayout.addWidget(modelInfoTitle, 0, QtCore.Qt.AlignTop)
        modelInfoTabLayout.addWidget(self.modelFileName, 0, QtCore.Qt.AlignTop)
        modelInfoTabLayout.addWidget(self.modelFileSize, 0, QtCore.Qt.AlignTop)
        modelInfoTabLayout.addWidget(self.modelEpoch, 0, QtCore.Qt.AlignTop)
        modelInfoTabLayout.addWidget(
            self.modelBatchSize, 0, QtCore.Qt.AlignTop)
        modelInfoTabLayout.addWidget(
            self.modelTrainValRatio, 0, QtCore.Qt.AlignTop)
        modelInfoTabLayout.addWidget(self.modelType, 0, QtCore.Qt.AlignTop)
        modelInfoTabLayout.addWidget(self.modelDelete, 1, QtCore.Qt.AlignBottom)
        modelInfoTab.setLayout(modelInfoTabLayout)
        modelInfoTab.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.MinimumExpanding
        )

        # create window widget
        window = QWidget()
        # Create layout for window and add widgets
        winLayout = QVBoxLayout()
        winLayout.addWidget(titleContainer, 0)
        window.setLayout(winLayout)

        # create tabs container
        tabs = QWidget()
        tabLayout = QHBoxLayout()
        tabLayout.addWidget(uploadsTab, 1)
        tabLayout.addWidget(modelInfoTab, 1)
        tabs.setLayout(tabLayout)

        # add tabs onto window
        winLayout.addWidget(tabs)

        # Set layout for window
        self.setLayout(winLayout)

    def goHome(self):
        self.controller.showLandingView()
