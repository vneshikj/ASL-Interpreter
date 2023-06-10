import os
import sys
from pathlib import Path
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QListView, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QSizePolicy, QStackedWidget, QFileDialog

dir = os.path.dirname(__file__)


class TestingView(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setStyleSheet(
            Path(os.path.join(dir, 'testingview.qss')).read_text())

        # Create label for title
        title = QLabel("Test Model/Make Predictions")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        # Create button to go back
        backButton = QToolButton()
        backButton.setIcon(QtGui.QIcon(
            "ui/styles/theme/primary/leftarrow.svg"))
        backButton.clicked.connect(self.controller.showLandingView)

        # Buttons to select use camera or load images
        cameraButton = QPushButton("Capture")
        cameraButton.clicked.connect(self.goToCamera)
        filesButton = QPushButton("Load Images")
        filesButton.clicked.connect(self.goToFiles)
        self.modelsButton = QPushButton("Select Model")
        self.modelsButton.clicked.connect(self.goToModels)

        # Create container widget
        container = QWidget()
        container.setStyleSheet(
            "background-color: #53433E; border-radius: 4px;")

        # Add widgets to container layout
        titleContainer = QWidget()

        # Create a layout for the breadcrumb menu
        breadcrumbLayout = QHBoxLayout()
        breadcrumbLayout.addWidget(cameraButton)
        breadcrumbLayout.addWidget(filesButton)
        breadcrumbLayout.addWidget(self.modelsButton)
        breadcrumbLayout.addStretch()

        # Camera Widgets & layouts
        cameraControls = QVBoxLayout()
        self.openCameraButton = QPushButton("Use Camera")
        self.openCameraButton.setIcon(QtGui.QIcon())
        self.captureButton = QPushButton("Capture")
        self.captureButton.hide()
        cameraControls.addWidget(self.openCameraButton)
        cameraControls.addWidget(self.captureButton)
        self.cameraFeed = QLabel()
        self.cameraFeed.setScaledContents(True)
        self.cameraFeed.hide()
        # Layout for Camera File manager
        cameraFileManagerLayout = QVBoxLayout()
        self.cameraFileManagerTitle = QLabel("Last Capture")
        self.lastCapture = QLabel()
        self.lastCapture.setScaledContents(True)
        self.lastPrediction = QLabel("--")
        self.cameraFileManagerTitle.setAlignment(QtCore.Qt.AlignTop)
        self.cameraFileManagerTitle.setStyleSheet("font-size: 20px;")

        # Adding widgets to vertical layout
        cameraFileManagerLayout.addWidget(self.cameraFileManagerTitle)
        cameraFileManagerLayout.addWidget(self.lastCapture)
        cameraFileManagerLayout.addWidget(self.lastPrediction)
        cameraFileManagerLayout.addStretch()

        # Container for files manager
        cameraFileManagerContainer = QWidget()
        cameraFileManagerContainer.setLayout(cameraFileManagerLayout)
        cameraFileManagerContainer.setStyleSheet(
            "background-color:#FFB599; color: #7F2B00 ")

        # Layout for Predictions container
        predictionsLayout = QVBoxLayout()
        predictionsLayoutTitle = QLabel("Predictions")
        predictionsLayoutAccuracy = QLabel("Accuracy: 100%")
        predictionsLayout.addStretch()

        predictionsLayout.addWidget(predictionsLayoutTitle)
        predictionsLayout.addWidget(predictionsLayoutAccuracy)

        # Creating container for predictions
        predictionsContainer = QWidget()
        predictionsContainer.setLayout(predictionsLayout)
        predictionsContainer.setStyleSheet(
            "background-color:#FFB599; color: #7F2B00 ")

        # Creating layout for button container
        buttonLayout = QHBoxLayout()
        changeDirectoryButton = QPushButton("Change Directory")
        buttonLayout.addWidget(changeDirectoryButton)

        # Creating container for buttons
        buttonContainer = QWidget()
        buttonContainer.setLayout(buttonLayout)
        buttonContainer.setStyleSheet(
            "background-color: #FFB599; color: #7F2B00")

        # Layout for camera option
        cameraLayout = QGridLayout()
        cameraLayout.addWidget(self.cameraFeed, 1, 0)
        cameraLayout.addLayout(cameraControls, 2,
                               0, QtCore.Qt.AlignVCenter)
        cameraLayout.addWidget(cameraFileManagerContainer,
                               1, 1)
        cameraLayout.addWidget(predictionsContainer, 0, 1)
        cameraLayout.addWidget(buttonContainer, 2, 1)
        cameraContainer = QWidget()
        cameraContainer.setLayout(cameraLayout)
        cameraContainer.setStyleSheet(
            'background-color: #53433E; border-radius: 4px;')

        filesLayout = QVBoxLayout()
        imagePredictionLayout = QGridLayout()
        self.currentImage = QLabel()
        self.currentImage.setScaledContents(True)
        self.currentImage.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.currentPrediction = QLabel("--")
        self.currentPrediction.setStyleSheet(
            "color: #53433E; font-size: 100px;")
        self.goNextPredictionButton = QPushButton("Next")
        self.goNextPredictionButton.setStyleSheet(
            "background-color: #53433E;")
        self.goBackPredictionButton = QPushButton("Back")
        self.goBackPredictionButton.setStyleSheet(
            "background-color: #53433E;")
        self.predictionLabel = QLabel("Prediction")
        self.predictionLabel.setStyleSheet('color: #53433E; font-size: 40px')
        navigatePredictionButtonContainer = QHBoxLayout()
        navigatePredictionButtonContainer.addStretch()
        navigatePredictionButtonContainer.addWidget(
            self.goBackPredictionButton, 0)
        navigatePredictionButtonContainer.addWidget(
            self.goNextPredictionButton, 0)
        imagePredictionLayout.addWidget(
            self.currentImage, 1, 0)
        imagePredictionLayout.addWidget(
            self.predictionLabel, 0, 1, QtCore.Qt.AlignTop)
        imagePredictionLayout.addWidget(
            self.currentPrediction, 1, 1)
        imagePredictionLayout.addLayout(
            navigatePredictionButtonContainer, 2, 2)
        imagePredictionLayout.setRowStretch(0, 0)
        imagePredictionLayout.setRowStretch(1, 2)
        imagePredictionLayout.setColumnStretch(2, 0)

        # Set the horizontal stretch factor of column 0 in imagePredictionLayout to 1
        imagePredictionLayout.setColumnStretch(1, 1)
        imagePredictionLayout.setColumnStretch(0, 2)

        self.predictButton = QPushButton("Predict")
        self.predictButton.setStyleSheet(
            "background-color:#53433E; ")
        self.loadImagesButton = QPushButton("Load Images")
        self.loadImagesButton.setStyleSheet(
            "background-color:#53433E; ")
        hBox = QHBoxLayout()
        hBox.addWidget(self.loadImagesButton)
        hBox.addWidget(self.predictButton)
        uploadsTab = QWidget()
        goToUploadButton = QPushButton("Go to upload")
        goToUploadButton.clicked.connect(self.goToUpload)
        goToUploadButton.setStyleSheet(
            "background-color: #FFB599;")
        uploadsTab.setStyleSheet(
            "background-color: #53433E; border-radius: 4px;")
        uploadsTabLayout = QVBoxLayout()
        uploadModelsTitle = QLabel("Uploaded Models")
        uploadModelsTitle.setStyleSheet("font-size: 24px; font-weight: bold;")
        uploadsTabLayout.addWidget(uploadModelsTitle, 0, QtCore.Qt.AlignTop)
        self.uploadedModels = QListView()
        uploadsTabLayout.addWidget(self.uploadedModels)
        uploadsTabLayout.addWidget(goToUploadButton)
        uploadsTab.setLayout(uploadsTabLayout)
        uploadsTab.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.MinimumExpanding
        )

        filesLayout.addLayout(imagePredictionLayout, 2)
        filesLayout.addLayout(hBox, 0)
        filesLayout.addStretch()

        # Container for files menu
        filesContainer = QWidget()
        filesContainer.setLayout(filesLayout)
        filesContainer.setStyleSheet(
            'background-color: #FFB599; border-radius: 4px;')

        # Layout for button
        filesButtonLayout = QHBoxLayout()
        uploadButton = QPushButton("Upload Files")
        filesButtonLayout.addWidget(uploadButton)

        # Container for button
        buttonContainer = QWidget()
        buttonContainer.setLayout(filesButtonLayout)

        # Create layout for container
        titleLayout = QHBoxLayout()
        containerLayout = QVBoxLayout()
        containerLayout.addWidget(QLabel("hate"))
        container.setLayout(containerLayout)
        container.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.MinimumExpanding
        )

        # add heading
        titleLayout.addWidget(backButton)
        titleLayout.addWidget(title)
        titleLayout.addStretch()

        titleContainer.setLayout(titleLayout)
        #
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(cameraContainer)
        self.stackedWidget.addWidget(filesContainer)
        self.stackedWidget.addWidget(uploadsTab)
        # Create layout for window
        layout = QVBoxLayout()
        layout.addWidget(titleContainer, 0, QtCore.Qt.AlignTop)
        layout.addLayout(breadcrumbLayout)
        layout.addWidget(self.stackedWidget, 1)

        # Set layout for window
        self.setLayout(layout)

    def goHome(self):
        self.controller.showLandingView()

    def goToUpload(self):
        self.controller.showLoadingView()

    def goToCamera(self):
        self.stackedWidget.setCurrentIndex(0)

    def goToFiles(self):
        self.stackedWidget.setCurrentIndex(1)

    def goToModels(self):
        self.stackedWidget.setCurrentIndex(2)
