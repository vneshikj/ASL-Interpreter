from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from ML.modules.predict import labelToLetter
import numpy as np
import traceback
from ui.utils.runnables import WebcamThread



class TestingController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self.webcamThread = WebcamThread()
        self.webcamThread.frameSignal.connect(self.updateFeed)
        self.view.captureButton.clicked.connect(self.captureImage)
        self.view.openCameraButton.clicked.connect(self.openCamera)
        self.view.loadImagesButton.clicked.connect(self.loadImagesFromFolder)
        self.view.predictButton.clicked.connect(self.startPredictions)
        self.view.goNextPredictionButton.clicked.connect(self.goToNextImage)
        self.view.goBackPredictionButton.clicked.connect(
            self.goToPreviousImage)
        self.view.uploadedModels.setModel(self.model.modelManager)
        self.view.uploadedModels.clicked.connect(self.updateModel)
        self.model.predictionsUpdated.connect(self.updatePredictions)
        self.view.goToModels()
        self.model.loading.connect(self.checkLoading)

    def killWebcam(self):
        """Function to allow the app to close gracefully on exit (prevent webcam thread from blocking the shutdown)"""
        self.webcamThread.stop()
        self.webcamThread.wait()

    def loadImagesFromFolder(self):
        """Obtains the list of files for prediction and notifies the model to load them for prediction"""
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            None, "Select Images", "", "Images (*.png *.jpg *.jpeg);;CSV files (*.csv)", options=options)
        if files:
            if len(files) == 1 and str(files[0])[-4:] == ".csv":
                self.model.loadCSV(str(files[0]))
                return
            self.model.loadImageGroup(files)

    def updatePredictions(self, prediction):
        """Parameters:
            @prediction: a dictionary containing two keys, "image" which is the image that was predicted and "prediction" which is the prediction result (string)
        This function notifies the view to update the current prediction result that is displayed 
        """
        image = prediction["image"]
        if isinstance(image, np.ndarray):
            # Convert the NumPy array to a QImage object
            height, width = image.shape
            bytesPerLine = width
            qImg = QImage(image.data, width, height,
                          bytesPerLine, QImage.Format_Grayscale8)

            # Create a QPixmap object from the QImage
            pixmap = QPixmap.fromImage(qImg)
        else:
            pixmap = QPixmap(image)

        self.view.currentImage.setPixmap(pixmap)
        self.view.predictionLabel.setText(f'Prediction ({prediction["accuracy"]:.4f})')
        self.view.currentPrediction.setText(labelToLetter(prediction["prediction"]))

    def openCamera(self):
        """Starts the webcam thread to allow image capture"""
        self.view.cameraFeed.show()
        self.view.openCameraButton.hide()
        self.view.captureButton.show()
        self.webcamThread.start()

    def captureImage(self):
        """Captures an image and makes a prediction based on that capture. There is also an option to save the image."""
        try:
            self.view.lastCapture.setPixmap(self.view.cameraFeed.pixmap())
            if self.model.modelToUse is not None:
                capturedImage = self.view.cameraFeed.pixmap().toImage()
                pred = self.model.predictOnCapture(capturedImage)
                self.view.lastPrediction.setText(labelToLetter(pred[0]))
                self.view.cameraFileManagerTitle.setText(f'Last Capture ({pred[1]:.4f})')
            filename, _ = QFileDialog.getSaveFileName(
                None, "Save Image", "", "Images (*.png *.xpm *.jpg)")
            if filename:
                self.view.cameraFeed.pixmap().save(filename)
        except:
            traceback.print_exc()
            print("error with capture")

    def updateModel(self, index):
        """Parameters:
            @index: the selected row on the models display
        Uses the selected row (representing a model) and sets it as the active model for prediction
        """
        self.model.updateModel(index.data(Qt.UserRole + 1))

    def goToNextImage(self):
        self.model.goNext()

    def goToPreviousImage(self):
        self.model.goBack()

    def startPredictions(self):
        self.model.startPredictions()

    def updateFeed(self, pixmap):
        """Parameters:
            @pixmap: a PyQt pixmap 
        Updates the webcamfeed with the latest frame from the webcam thread
        """
        self.view.cameraFeed.setPixmap(pixmap)

    def checkLoading(self, isLoading):
        """Parameters:
            @isLoading: a boolean signal emitted by the model based on status
        Disables changing and predicting while loading or predictions are in progress
        """
        if isLoading:
            self.view.modelsButton.setEnabled(False)
            self.view.predictButton.setEnabled(False)
            self.view.predictionLabel.setText("Loading...")
        else:
            self.view.modelsButton.setEnabled(True)
            self.view.predictButton.setEnabled(True)
            self.view.predictionLabel.setText("Prediction")
