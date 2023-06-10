from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QTableView, QWidget, QVBoxLayout, QAbstractItemView, QHeaderView, QPushButton, QStackedWidget, QToolButton, QSlider, QScrollArea, QMainWindow
from PyQt5.QtCore import QSize


class DatasetScrollArea(QScrollArea):
    def __init__(self):
        super().__init__()
        self.datasetTable = QTableView()
        self.datasetTable.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.datasetTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.datasetTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.datasetTable.setShowGrid(False)
        self.datasetTable.horizontalHeader().hide()
        self.datasetTable.verticalHeader().hide()
        self.datasetTable.verticalHeader().setDefaultSectionSize(100)
        self.datasetTable.horizontalHeader().setMinimumSectionSize(140)
        self.datasetTable.setStyleSheet(
            'background-color: #A08D86; border-radius: 4px;')
        self.setWidget(self.datasetTable)
        self.setWidgetResizable(True)
        self.setWidget(self.datasetTable)
        self.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QScrollBar:vertical {
                background: #A08D86;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #53433E;
                min-height: 20px;
            }
        """)

    def setTableModel(self, model):
        self.datasetTable.setModel(model)


class ResizableImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1, 1)
        self.setScaledContents(True)
        self.aspectRatio = 1.0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setAspectRatio(self, aspectRatio):
        self.aspectRatio = aspectRatio
        self.update()

    def sizeHint(self):
        width = self.height() * self.aspectRatio
        height = self.width() / self.aspectRatio
        return QSize(int(min(self.width(), width)), int(min(self.height(), height)))

    def resizeEvent(self, event):
        size = self.sizeHint()
        super().resizeEvent(event)
        if size.isValid():
            self.setFixedSize(size)
