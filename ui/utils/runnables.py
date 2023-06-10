from PyQt5.QtCore import QObject, pyqtSignal, QThread, QRunnable, QThreadPool, Qt, QRect, QSize, QByteArray
from PyQt5.QtGui import QImage, QPixmap
import cv2

# for previewing dataset


class ConvertCsvTask(QRunnable):

    def __init__(self, model, isCsv):
        super().__init__()
        self.model = model
        self.isCsv = isCsv

    def run(self):
        if self.isCsv:
            self.model.convertCSVToNumpy()
        else:
            self.model.convertFolderToNumpy()


class WebcamThread(QThread):
    frameSignal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.stopped = False

    def run(self):
        cap = cv2.VideoCapture(0)
        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = QPixmap.fromImage(convertToQtFormat)
                self.frameSignal.emit(p)
        cap.release()

    def stop(self):
        self.stopped = True
