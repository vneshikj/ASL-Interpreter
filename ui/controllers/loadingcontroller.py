from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QFileDialog
import torch
import os

class LoadController(QObject):
    """Controller for loading screen"""
    def __init__(self, model, view):
        """Parameters
        @model: model for loading view
        @view: view for loading screen"""
        super().__init__()
        self.model = model
        self.view = view
        self.view.loadModelButton.clicked.connect(self.addModel)
        self.view.uploadedModels.clicked.connect(self.displayModelInfo)
        self.view.modelDelete.clicked.connect(self.deleteSelectedModel)
        self.selectedIndex = None

    def goHome(self):
        """Click to go back to landing view"""
        self.view.controller.showLandingView()

    def addModel(self):
        """Add a model to be used for training/ testing"""
        try:
            options = QFileDialog.Options()
            fileFilter = "PyTorch files (*.pt *.pth);;All files (*)"
            filePath, _ = QFileDialog.getOpenFileName(
                None, "Open PyTorch file", "", fileFilter, options=options)
            if filePath:

                self.directory = filePath
                modelInfo = torch.load(filePath)
                if modelInfo.get("model_state_dict") is not None:
                    modelInfo["metadata"]["name"] = os.path.basename(filePath)
                    modelInfo["metadata"]["file_size"] = str(int(
                        os.path.getsize(filePath)/1024)) + "KB"  # in kb
                    self.model.append(modelInfo)
                    self.view.uploadedModels.setModel(self.model)
        except:
            print("incorrect file format")
            return
        
    def displayModelInfo(self, index):
        """Displays the model info
        @index: the index where the model is located"""
        data = index.data(Qt.UserRole)
        self.currentIndex = self.view.uploadedModels.currentIndex()

        print(data)
        self.view.modelFileName.setText(f'Filename: {data["name"]}')
        self.view.modelFileSize.setText(f'File Size: {data["file_size"]}')
        self.view.modelEpoch.setText(f'Epoch: {data["num_epochs"]}')
        self.view.modelBatchSize.setText(f'Size: {data["batch_size"]}')
        self.view.modelTrainValRatio.setText(f'Train Ratio: {data["train_ratio"]}')
        self.view.modelType.setText(f'Type: {data["architecture"]}')

    def deleteSelectedModel(self):
        """Deletes the selected model"""
        if self.currentIndex is None:
            return
        self.view.modelFileName.setText(f'Filename: ')
        self.view.modelFileSize.setText(f'File Size:')
        self.view.modelEpoch.setText(f'Epoch: ')
        self.view.modelBatchSize.setText(f'Size: ')
        self.view.modelTrainValRatio.setText(f'Train Ratio: ')
        self.view.modelType.setText(f'Type:')
        self.model.remove(self.currentIndex.row())
        self.currentIndex = None

    def updateViewDirectory(self):
        """Updates the directory where the model is located"""
        self.view.loadModelButton.setText(self.model.getDirectory())
