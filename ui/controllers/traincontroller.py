import re
from PyQt5.QtCore import QObject, QTimer, QThreadPool
from PyQt5.QtWidgets import QFileDialog
from ui.utils.runnables import ConvertCsvTask
from ui.views.previewview.previewview import PreviewView
from ui.models.previewmodel import PreviewModel
from ui.controllers.previewcontroller import PreviewController


class TrainController(QObject):
    """Train Controller class. This coordinates/ calls functions 
    for training when needed."""
    def __init__(self, model, view):
        """Parameters
            @model: the model for the training view
            @view: the view for the training
        """
        super().__init__()
        self.model = model
        self.view = view
        self.displayed = []
        self.images = []
        self.waitingForTrainStatus = False
        self.imageLoaderThread = None
        self.remainingLoadTimer = QTimer()
        self.remainingLoadTimer.timeout.connect(self.updateRemainingLoadTime)
        self.currentLoadTime = 60
        self.previewModel = PreviewModel()
        self.view.datasetPreview.setTableModel(self.previewModel)
        self.view.previewDatasetButton.clicked.connect(self.goToPreviewWindow)
        self.view.openFolderButton.clicked.connect(self.updateDirectory)
        self.view.openCSVButton.clicked.connect(self.updateDirectoryCSV)
        self.view.epochSlider.valueChanged.connect(self.updateEpoch)
        self.view.batchSizeSlider.valueChanged.connect(
            self.updateBatchSize)
        self.view.trainSlider.valueChanged.connect(self.updateTrainRatio)
        self.view.trainSlider.valueChanged.connect(self.updateTrainValLabel)
        self.view.presetModelsDropdown.currentIndexChanged.connect(
            self.usePresetModel)
        self.view.presetModelsDropdown.addItems(self.model.getPresetModels())
        self.view.trainingButton.clicked.connect(self.toggleTraining)
        self.view.saveButton.clicked.connect(self.saveTrainedModel)
        self.view.stopLoadingButton.clicked.connect(self.stopLoading)
        self.model.directoryUpdated.connect(self.updateViewDirectory)
        self.model.percentageChanged.connect(self.updatePercentage)
        self.model.logsChanged.connect(self.updateLogs)
        self.model.trainingCancelled.connect(self.onTrainingCancelled)
        self.model.trainingStarted.connect(self.onTrainingStarted)
        self.model.trainingComplete.connect(self.onTrainStatusChanged)
        self.model.remainingTimeChanged.connect(self.updateRemainingTime)
        self.model.datasetLoaded.connect(self.updateLoadingStatus)
        self.model.splitDone.connect(self.updateTrainValDisplay)

    def goHome(self):
        """Goes back to landing page."""
        self.view.controller.showLandingView()

    def toggleTraining(self):
        """Tells model to start training process"""
        if not self.waitingForTrainStatus:
            if not self.model.isTraining:
                self.model.startTraining()
                self.view.trainingButton.setText("Stop")
            else:
                self.model.stopTraining()
                self.view.trainingButton.setText("Start")

    def saveTrainedModel(self):
        """Saves the trained model. Communicates with the model class."""
        filePath, _ = QFileDialog.getSaveFileName(
            None, "Save As", "", "PyTorch Models (*.pt), PyTorch Models (*.pth)")
        if filePath:
            print("Model saved at:" + filePath)
            self.model.saveModel(filePath)

    def onTrainStatusChanged(self, isComplete):
        """When the training process has finished, activate save button."""
        if isComplete:
            self.view.saveButton.setDisabled(False)
        else:
            self.view.saveButton.setDisabled(True)

    def stopLoading(self):
        self.view.datasetInformation.setText("Stopped")
        self.view.loadingIcon.hide()
        self.view.stopLoadingButton.hide()

    def usePresetModel(self):
        """Uses the preset model, selected from the dropdown"""
        self.model.usePresetModel(self.view.presetModelsDropdown.currentText())

    def updatePercentage(self, percentage):
        """Updates the training percentage indicating process"""
        self.view.trainProgress.setValue(int(percentage))

    def updateRemainingLoadTime(self):
        minutes = self.currentLoadTime // 60
        seconds = self.currentLoadTime % 60
        self.view.remainingLoadingTime.setText(f"{minutes:02d} M {seconds:02d} S")
        self.currentLoadTime -= 1

    def updateLogs(self, log):
        """Updates the training logs for loss"""
        self.view.logs.setText(log)
        matches = re.findall(r'\d+\.\d+', log)
        lastFourFloats = [float(f) for f in matches[-4:] if float(f) <= 1]
        if len(lastFourFloats) != 4:
            return
        self.view.currentLoss.setText(str(lastFourFloats[2]))
        self.view.currentAccuracy.setText(str(lastFourFloats[3]))

    def updateTrainRatio(self):
        """Updates the training/validation split on training data based on the slider."""
        self.model.updateTrainRatio(self.view.trainSlider.value())

    def updateTrainValDisplay(self, trainVal):
        """Updates the text indicating the split in the dataset between
            training and validation data.
             @trainVal: Array containing the percentage of training and validation """
        self.view.trainValRatio.setText(
            f'Train/Validate Split ({trainVal[0]}/{trainVal[1]})')

    def goToPreviewWindow(self):
        """Instantiates a preview window to preview datasets before selection."""
        # instantiate preview window
        self.previewView = PreviewView()
        self.previewModel = PreviewModel()

        self.previewController = PreviewController(
            self.previewView, self.previewModel, self.model.directory)
        if (self.model.directory == ""):
            self.view.stackedWidget.setCurrentIndex(0)  # add new dataset

    def updateDirectory(self):
        """Updates the directory where the dataset is located. Calls functions
        from trainModel"""
        self.model.updateDirectory()
        self.previewModel.directory = self.model.directory
        self.view.stopLoadingButton.show()
        task = ConvertCsvTask(self.previewModel, isCsv=False)
        # Submit the task to a thread pool to run on another thread
        threadPool = QThreadPool.globalInstance()
        threadPool.start(task)

    def updateDirectoryCSV(self):
        """Updates the directory where the CSV dataset is located. Calls functions
        from trainModel."""
        self.model.updateDirectoryCSV()
        self.previewModel.directory = self.model.directory
        self.view.stopLoadingButton.show()
        task = ConvertCsvTask(self.previewModel, isCsv=True)
        # Submit the task to a thread pool to run on another thread
        threadPool = QThreadPool.globalInstance()
        threadPool.start(task)

    def updateViewDirectory(self):
        """Shows the directory where the dataset is located on UI"""
        self.view.currentDatasetLabel.setText(self.model.getDirectory())

    def modelSelect(self):
        """Selects the model based on the dropdown"""
        self.view.addModelButton.setText(self.model.getDirectory())

    def updateEpoch(self):
        """Updates the epoch number to be seen on training screen"""
        epochValue = self.view.epochSlider.value()
        self.view.epochValueLabel.setText(str(epochValue))
        self.model.updateHyperparameters("Epochs", epochValue)

    def updateBatchSize(self):
        """Updates batch size hyperparameter based on the batchSizeSlider"""
        batchSizeValue = self.view.batchSizeSlider.value()
        self.view.batchSizeValueLabel.setText(
            str(batchSizeValue))
        self.model.updateHyperparameters("BatchSize", batchSizeValue)

    def updateTrainValLabel(self):
        """Updates the traininng/ validation split percentage"""
        self.view.trainValueLabel.setText(
            str(self.view.trainSlider.value()) + "%")

    def onTrainingStarted(self):
        """Changes the start training button to a stop training button"""
        self.view.trainingButton.setText("Stop")

    def onTrainingCancelled(self):
        """Changes the stop training button to a start training button"""
        self.view.trainingButton.setText("Start")

    def updateRemainingTime(self, time):
        """Updates the remaining time needed for training
        @time: int: the time left for training needed in seconds"""
        min, sec = divmod(time, 60)
        self.view.remainingTime.setText(f"{int(min)}M {int(sec)}S")

    def updateLoadingStatus(self, isLoaded):
        """Updates the loading dataset status. Once loaded, changes it to loaded
        @isLoaded: boolean; signal"""
        if isLoaded:
            self.view.loadingIcon.hide()
            self.view.datasetInformation.setText("Loaded")
            self.view.stopLoadingButton.hide()
            self.view.remainingLoadingTime.hide()
            self.remainingLoadTimer.stop()
        else:
            self.view.loadingIcon.show()
            self.view.datasetInformation.setText("Loading...")
            self.currentLoadTime = 47
            self.view.remainingLoadingTime.show()
            self.remainingLoadTimer.start(1000)
