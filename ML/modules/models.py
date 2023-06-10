from torch.nn import Module
from torch.nn import Conv2d  # convolutional layers
from torch.nn import Linear  # fully connected layers
# applies 2d max-pooling to reduce spatial dimensions of input volume
from torch.nn import MaxPool2d
from torch.nn import ReLU  # ReLU activation function
from torch.nn import LogSoftmax  # returns predicted probabilities of each class
from torch.nn import AvgPool2d
from torch.nn import Dropout
from torch.nn import BatchNorm2d
from torch.nn import Sequential, AdaptiveAvgPool2d
# flattens output of a multidimensional volume so we can apply fully connected layers
from torch import flatten

class LeNet(Module):
    """ Parameters
        @numChannels: int, number of channels (1 for greyscale and 3 for RGB)
        @classes: int, number of different classes/ characters there are in the dataset to be generated
        @returns: prediction of inputted image
    """
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels,
                            out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout1 = Dropout(p=0.25)

        # initialize second set of layers
        self.conv2 = Conv2d(
            in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout2 = Dropout(p=0.25)

        # initialize first set of FC => RELU layers
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()
        self.dropout3 = Dropout(p=0.5)

        # initialize softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV, RELU and POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # pass input through second set of layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # flatten output from previous layer and pass it through our only set of FC -> RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # pass output to our softmax classifier to get predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)

        # return output predictions
        return output



class AlexNet(Module):
    """ Parameters
        @numChannels: int, number of channels (1 for greyscale and 3 for RGB)
        @classes: int, number of different classes/ characters there are in the dataset to be generated
        @returns: prediction of inputted image
    """
    def __init__(self, numChannels, classes):
        super(AlexNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels,
                            out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.avgpool1 = AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of layers
        self.conv2 = Conv2d(
            in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.avgpool2 = AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # third set of layers
        self.conv3 = Conv2d(
            in_channels=50, out_channels=100, kernel_size=(4, 4))
        self.relu3 = ReLU()

        # fourth set of layers
        self.conv4 = Conv2d(
            in_channels=100, out_channels=250, kernel_size=(1, 1))
        self.relu4 = ReLU()

        # initialize first set of Fully Connected => RELU layers
        self.fc1 = Linear(in_features=250, out_features=50)
        self.relu6 = ReLU()

        # initialize softmax classifier and
        # second set of fully connected -> RELu layers
        self.fc2 = Linear(in_features=50, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)

        out = self.logSoftmax(x)
        return out


class VGG(Module):
    """ Parameters
        @numChannels: int, number of channels (1 for greyscale and 3 for RGB)
        @classes: int, number of different classes/ characters there are in the dataset to be generated
        @returns: prediction of inputted image
    """
    def __init__(self, numChannels, classes):
        super(VGG, self).__init__()
        self.features = Sequential(
            Conv2d(in_channels= numChannels, out_channels=64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(128, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            #MaxPool2d(kernel_size=2, stride=2),
            #Conv2d(512, 512, kernel_size=3, padding=1),
            #ReLU(inplace=True),
            #Conv2d(512, 512, kernel_size=3, padding=1),
            #ReLU(inplace=True),
            #Conv2d(512, 512, kernel_size=3, padding=1),
            #ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = AdaptiveAvgPool2d((7, 7))
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 1024),
            ReLU(inplace=True),
            #Dropout(),
            Linear(1024, 500),
            ReLU(inplace=True),
            #Dropout(),
            Linear(500, classes),
        )
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        out = self.logSoftmax(x)
        return out

