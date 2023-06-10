import sys
import os
import PyQt5
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from ui.app import App

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_UseHighDpiPixmaps, True)

# Create the application and run the main event loop
if __name__ == "__main__":
    print("App Starting Up...")
    dir = os.path.dirname(__file__)
    app = QApplication(sys.argv)
    ex = App()
    app.setStyleSheet(
        Path(os.path.join(dir, 'ui', 'styles', 'default_theme.qss')).read_text())
    app.quit()
    sys.exit(app.exec_())
