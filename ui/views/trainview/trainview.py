import os
from pathlib import Path
from ui.utils.components import DatasetScrollArea
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QProgressBar, QComboBox, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QToolButton, QSlider, QScrollArea, QMainWindow    
dir = os.path.dirname(__file__)


    
class TrainView(QWidget):
    """Class containing UI for training, dataset, model select, hyperparameters screen."""
    def __init__(self, controller):
        """Parameters
        @controller: the controller for this view"""
        # Folder Select Button
        super().__init__()
        self.controller = controller
        self.setStyleSheet(
            Path(os.path.join(dir, 'trainview.qss')).read_text())

        # Create the breadcrumb menu
        datasetButton = QPushButton("Dataset")
        modelSelectButton = QPushButton("Model Select")
        hyperparametersButton = QPushButton("Hyperparameters")
        trainButton = QPushButton("Train")

        # Define onclicks
        datasetButton.clicked.connect(self.goToDataset)
        modelSelectButton.clicked.connect(self.goToModelSelect)
        hyperparametersButton.clicked.connect(self.goToHP)
        trainButton.clicked.connect(self.goToTrain)

        # Create the back button
        backButton = QToolButton()
        backButton.setIcon(QtGui.QIcon(
            "ui/styles/theme/primary/leftarrow.svg"))
        backButton.clicked.connect(self.controller.showLandingView)

        # Create a layout for the breadcrumb menu and back button
        breadcrumbLayout = QHBoxLayout()
        breadcrumbLayout.addWidget(datasetButton)
        breadcrumbLayout.addWidget(hyperparametersButton)
        breadcrumbLayout.addWidget(modelSelectButton)
        breadcrumbLayout.addWidget(trainButton)
        breadcrumbLayout.addStretch()

        # View Widgets
        title = QLabel("Train")
        title.setStyleSheet('font-size: 24px;  font-weight: bold;')
        titleLayout = QHBoxLayout()
        titleLayout.addWidget(backButton)
        titleLayout.addWidget(title)
        titleLayout.addStretch()

        # Dataset widgets
        self.previewDatasetButton = QPushButton("Preview Dataset")
        self.previewDatasetButton.setStyleSheet(
            'background-color: #A08D86; font-size:10px')

        self.openFolderButton = QPushButton("Open Dataset Folder")
        self.openFolderButton.setStyleSheet(
            'background-color: #A08D86; font-size:10px')
        self.openCSVButton = QPushButton("Open CSV (MNIST Format)")
        self.openCSVButton.setStyleSheet(
            'background-color: #A08D86; font-size:10px')
        self.currentDatasetLabel = QLabel("../")
        self.currentDatasetLabel.setStyleSheet(
            'font-size:10px')

        datasetLoadingSpinner = QMovie('ui/assets/loadingicon.gif')
        datasetLoadingSpinner.start()
        self.loadingIcon = QLabel()
        self.loadingIcon.setMovie(datasetLoadingSpinner)
        self.loadingIcon.hide()
        self.datasetInformation = QLabel("Not Loaded")
        datasetInformationLayout = QVBoxLayout()
        datasetInformationLayout.addWidget(self.loadingIcon, QtCore.Qt.AlignHCenter)
        datasetInformationLayout.addWidget(self.datasetInformation)
        datasetInformationContainer = QWidget()
        self.stopLoadingButton = QPushButton("Stop")
        self.stopLoadingButton.hide()
        self.remainingLoadingTime = QLabel("-- M -- S")
        self.remainingLoadingTime.hide()
        datasetInformationLayout.addWidget(self.remainingLoadingTime)
        datasetInformationLayout.addWidget(self.stopLoadingButton)
        datasetInformationContainer.setLayout(datasetInformationLayout)
        datasetInformationContainer.setStyleSheet('background-color: #A08D86')

        self.trainSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.trainSlider.setRange(1, 99)
        self.trainSlider.setSingleStep(2)
        self.trainSlider.setValue(50)
        self.trainValueLabel = QLabel("50%")
        self.datasetPreview = DatasetScrollArea()

        # Hyper Parameter widgetstrainview
        self.epochSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.epochSlider.setRange(1, 50)
        self.epochSlider.setValue(5)
        self.epochSlider.setSingleStep(1)
        self.epochValueLabel = QLabel(str(self.epochSlider.value()))

        self.batchSizeSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.batchSizeSlider.setRange(1, 1000)
        self.batchSizeSlider.setValue(32)
        self.batchSizeSlider.setSingleStep(1)
        self.batchSizeValueLabel = QLabel(str(self.batchSizeSlider.value()))

        self.learningRateSlider = QSlider(QtCore.Qt.Orientation.Horizontal)

        # Training Widgets
        self.trainProgress = QProgressBar()
        self.trainProgress.setValue(0)
        self.trainProgress.setStyleSheet('background-color: #A08D86')
        self.trainingButton = QPushButton("Start")
        self.trainingButton.setFixedHeight(50)
        self.trainingButton.setStyleSheet('background-color: #A08D86')
        # -> Loss and Accuracy Displays
        timeContainer = QVBoxLayout()
        lossContainer = QVBoxLayout()
        accuracyContainer = QVBoxLayout()

        stats = QWidget()
        statContainer = QHBoxLayout(stats)
        self.currentLoss = QLabel("--")
        self.currentAccuracy = QLabel("--")
        self.remainingTime = QLabel("-- M -- S")
        self.currentLoss.setObjectName("stat")
        self.currentAccuracy.setObjectName("stat")
        self.remainingTime.setObjectName("stat")
        self.logs = QLabel("")
        self.saveButton = QPushButton("Save")
        self.saveButton.setStyleSheet("background-color:#FFB599;")
        self.saveButton.setDisabled(True)
        saveLayout = QHBoxLayout()
        saveLayout.addStretch()
        saveLayout.addWidget(self.saveButton)

        logContainer = QScrollArea()
        logContainer.setWidgetResizable(True)
        logContainer.setWidget(self.logs)
        logContainer.setStyleSheet('background-color: #A08D86; padding: 10px;')

        timeContainer.addWidget(QLabel("Time Remaining"))
        timeContainer.addWidget(self.remainingTime)
        timeContainer.addStretch()
        timeContainer.setSpacing(0)

        accuracyContainer.addWidget(QLabel("Accuracy"))
        accuracyContainer.addWidget(self.currentAccuracy)
        accuracyContainer.addStretch()
        accuracyContainer.setSpacing(0)

        lossContainer.addWidget(QLabel("Loss"))
        lossContainer.addWidget(self.currentLoss)
        lossContainer.addStretch()
        lossContainer.setSpacing(0)

        # Model Select Widgets
        modelSelWindow = QWidget()
        modelSelWindowLayout = QVBoxLayout()
        self.presetModelsDropdown = QComboBox()
        modelSelWindowLayout.addWidget(self.presetModelsDropdown)
        modelSelWindow.setLayout(modelSelWindowLayout)

        # container options
        statContainer.addWidget(self.trainingButton)
        statContainer.addStretch()
        statContainer.addLayout(timeContainer)
        statContainer.addLayout(lossContainer)
        statContainer.addLayout(accuracyContainer)
        statContainer.setSpacing(30)
        stats.setMaximumHeight(100)
        # Create the screens for the stacked widget
        datasetScreen = QVBoxLayout()
        hyperparametersScreen = QVBoxLayout()
        trainScreen = QVBoxLayout()
        modelSelScreen = QVBoxLayout()

        # Add Widgets to the Screens:

        # ---------------------------------
        # Dataset Screen
        datasetTitle = QLabel("Dataset")
        datasetTitle.setObjectName("sectionTitle")
        datasetScreen.addWidget(datasetTitle, 0, QtCore.Qt.AlignTop)
        datasetScreen.addWidget(self.openFolderButton)
        datasetScreen.addWidget(self.openCSVButton)
        datasetScreen.addWidget(self.previewDatasetButton)
        datasetScreen.addWidget(self.currentDatasetLabel,
                                0, QtCore.Qt.AlignCenter)
        datasetScreen.addWidget(
            QLabel("Dataset Information"), 0, QtCore.Qt.AlignTop)
        datasetScreen.addWidget(datasetInformationContainer)
        self.trainValRatio = QLabel("Train/Validate Split")
        datasetScreen.addWidget(self.trainValRatio)
        datasetScreen.addWidget(self.trainValueLabel)
        datasetScreen.addWidget(self.trainSlider)
        datasetScreen.addWidget(self.datasetPreview, 1)
        datasetScreen.addStretch()

        # Hyper Parameters Screen
        hpTitle = QLabel("Hyper Parameters")
        hpTitle.setObjectName("sectionTitle")
        hyperparametersScreen.addWidget(
            hpTitle, 0, QtCore.Qt.AlignTop)
        hyperparametersScreen.addWidget(QLabel("Epochs"))
        hyperparametersScreen.addWidget(self.epochValueLabel)
        hyperparametersScreen.addWidget(self.epochSlider)  # Epoch Slider
        hyperparametersScreen.addWidget(QLabel("Batch Size"))
        hyperparametersScreen.addWidget(self.batchSizeValueLabel)
        hyperparametersScreen.addWidget(
            self.batchSizeSlider)  # Batch Size Slider
        # hyperparametersScreen.addWidget(QLabel("Learning Rate"))
        # hyperparametersScreen.addWidget(self.learningRateSlider) # Learning Rate Slider
        hyperparametersScreen.addStretch()

        # Training Screen
        trainTitle = QLabel("Training")
        trainTitle.setObjectName("sectionTitle")
        trainScreen.addWidget(trainTitle, 0, QtCore.Qt.AlignTop)
        trainScreen.addWidget(stats, 0, QtCore.Qt.AlignTop)
        trainScreen.addWidget(self.trainProgress, 0, QtCore.Qt.AlignTop)
        trainScreen.addWidget(logContainer, 1)
        trainScreen.addLayout(saveLayout, 0)
        trainScreen.addStretch()
        # ----------------------------------

        # Model Select Screen
        modelSelTitle = QLabel("Model Select")
        modelSelTitle.setObjectName("sectionTitle")
        modelSelScreen.addWidget(modelSelTitle, 0, QtCore.Qt.AlignTop)
        modelSelScreen.addWidget(modelSelWindow, 1, QtCore.Qt.AlignTop)

        # Create the containers for the stacked widget
        datasetContainer = QWidget()
        hyperparametersContainer = QWidget()
        trainContainer = QWidget()
        modelSelectContainer = QWidget()

        # Screen to container
        datasetContainer.setLayout(datasetScreen)
        datasetContainer.setMaximumWidth(1400)

        hyperparametersContainer.setLayout(hyperparametersScreen)

        trainContainer.setLayout(trainScreen)

        modelSelectContainer.setLayout(modelSelScreen)

        # Set BG for all containers
        datasetContainer.setStyleSheet(
            'background-color: #53433E; border-radius: 4px;')
        hyperparametersContainer.setStyleSheet(
            'background-color: #53433E; border-radius: 4px;')
        trainContainer.setStyleSheet(
            'background-color: #53433E; border-radius: 4px;')
        modelSelectContainer.setStyleSheet(
            'background-color: #53433E; border-radius: 4px;')

        # Create the stacked widget and add the screens
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(datasetContainer)
        self.stackedWidget.addWidget(hyperparametersContainer)
        self.stackedWidget.addWidget(modelSelectContainer)
        self.stackedWidget.addWidget(trainContainer)

        # Create a layout for the stacked widget
        stackedLayout = QVBoxLayout()
        stackedLayout.addLayout(titleLayout)
        stackedLayout.addLayout(breadcrumbLayout)
        stackedLayout.addWidget(self.stackedWidget)

        # Set the layout for the widget
        self.setLayout(stackedLayout)

    def back(self):
        """Click to return to previous screen"""
        currentIndex = self.stackedWidget.currentIndex()
        if currentIndex > 0:
            self.stackedWidget.setCurrentIndex(currentIndex - 1)

    def goToDataset(self):
        """Go to dataset information screen"""
        self.stackedWidget.setCurrentIndex(0)

    def goToHP(self):
        """Go to Hyper Parameters screen"""
        self.stackedWidget.setCurrentIndex(1)

    def goToModelSelect(self):
        """Go to model select screen"""
        self.stackedWidget.setCurrentIndex(2)

    def goToTrain(self):
        """Go to training screen"""
        self.stackedWidget.setCurrentIndex(3)
        

