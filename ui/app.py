from ui.controller import MyController
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goats ML")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.controller = MyController(self.layout)
        self.showLandingView()
        self.show()

    def showLandingView(self):
        self.controller.showLandingView()

    def closeEvent(self, event):
        self.controller.testingController.killWebcam()
        event.accept()
