import os
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
# Define the View

dir = os.path.dirname(__file__)


class LandingView(QtWidgets.QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        # Set up the user interface
        self.setStyleSheet(Path(os.path.join(dir, 'landingview.qss')).read_text())
        welcomeLabel = QtWidgets.QLabel("Welcome to Goats ML")
        welcomeLabel.setAlignment(QtCore.Qt.AlignCenter)

        instructionLabel = QtWidgets.QLabel("Choose an option")
        instructionLabel.setAlignment(QtCore.Qt.AlignCenter)

        # 1. Train Button
        self.buttonTrain = QtWidgets.QPushButton()
        self.buttonTrain.setIcon(QtGui.QIcon('ui/assets/trainicon.svg'))
        self.buttonTrain.setIconSize(QtCore.QSize(300, 300))
        self.buttonTrain.clicked.connect(self.showTrain)

        # 2. Load Button
        self.buttonLoad = QtWidgets.QPushButton()
        self.buttonLoad.setIcon(QtGui.QIcon('ui/assets/uploadicon.svg'))
        self.buttonLoad.setIconSize(QtCore.QSize(300, 300))
        self.buttonLoad.clicked.connect(self.showLoad)

        # 3. Test Button
        self.buttonTest = QtWidgets.QPushButton()
        self.buttonTest.setIcon(QtGui.QIcon('ui/assets/cameraicon.svg'))
        self.buttonTest.setIconSize(QtCore.QSize(300, 300))
        self.buttonTest.clicked.connect(self.showTest)

        buttonGrid = QtWidgets.QGridLayout()

        containerGrid = QtWidgets.QGridLayout()
        container = QtWidgets.QWidget()
        hBox = QtWidgets.QHBoxLayout()
        vBox = QtWidgets.QVBoxLayout()

        # 1x3 grid settings
        buttonGrid.setSpacing(20)
        buttonGrid.addWidget(self.buttonTrain, 0, 1)
        buttonGrid.addWidget(self.buttonLoad, 0, 2)
        buttonGrid.addWidget(self.buttonTest, 0, 3)
        buttonGrid.setRowStretch(0, 1)
        buttonGrid.setColumnStretch(1, 1)
        buttonGrid.setColumnStretch(2, 1)
        buttonGrid.setColumnStretch(3, 1)

        container.setLayout(buttonGrid)

        # Allow buttons to resize cred:https://stackoverflow.com/questions/22870300/pyqt-issues-with-buttons-and-qgridlayout
        self.buttonTrain.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        self.buttonLoad.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        self.buttonTest.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)

        containerGrid.spacerItem()
        hBox.addWidget(container)
        containerGrid.addWidget(welcomeLabel, 0, 0, QtCore.Qt.AlignBottom)
        containerGrid.addWidget(container, 1, 0)
        containerGrid.addWidget(instructionLabel, 2, 0, QtCore.Qt.AlignTop)

        vBox.addWidget(welcomeLabel)
        vBox.addWidget(container)
        vBox.addWidget(instructionLabel)
        self.setLayout(vBox)

    def setData(self, data):
        raise Exception("not implemented")

    def connect_controller(self, controller):
        self.controller = controller

    def showTrain(self):
        self.controller.showTrainView()

    def showLoad(self):
        self.controller.showLoadingView()

    def showTest(self):
        self.controller.showTestingView()