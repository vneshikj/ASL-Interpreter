from PyQt5.QtCore import QAbstractListModel, QModelIndex, Qt
from PyQt5.QtWidgets import QListView
import torch


class LoadModel(QAbstractListModel):
    """Class for loading screen model"""
    def __init__(self, data=None):
        """Parameters:
        @data: the dataset"""
        super().__init__()
        self._data = data or []

    def rowCount(self, parent=QModelIndex()):
        """Counts the rows in dataset"""
        return len(self._data)

    def data(self, index, role=Qt.DisplayRole):
        """gets the image data from a specific index"""
        if role == Qt.DisplayRole:
            # only name
            return str(self._data[index.row()]["metadata"]["name"])
        elif role == Qt.UserRole:
            return self._data[index.row()]["metadata"]  # only metadata
        elif role == Qt.UserRole + 1:
            return self._data[index.row()]  # whole model
        return None

    def append(self, item):
        """Appends additional data onto existing dataset
        @item: additional data to be appended"""
        self.beginInsertRows(QModelIndex(), len(self._data), len(self._data))
        self._data.append(item)
        self.endInsertRows()

    def remove(self, index):
        """Removes data from dataset
        @index: int; the index to be removed"""
        self.beginRemoveRows(QModelIndex(), index, index)
        del self._data[index]
        self.endRemoveRows()
