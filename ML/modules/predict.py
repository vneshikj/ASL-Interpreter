# set the numpy seed for better reproducibilitypredict
from ML.modules.models import LeNet, AlexNet, VGG
from PIL import Image
from torchvision import transforms
import pandas as pd
import cv2
import torch
import imutils
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np

from ML.modules.ConvertDataset import ConvertDataset
np.random.seed(48)
# import the necessary packages


def mnistToNpArray(csvPath):
    """
    Takes in a CSV file to convert its rows and columns into 28x28 images 
    Parameters:
    @csvPath: String; the filepath of your CSV file
    @returns: array of resized images from the CSV file
    """
    # Load the MNIST dataset from CSV file
    df = pd.read_csv(csvPath, skiprows=[0])
    imArray = df.iloc[:, 1:].values
    imArray = imArray.astype(np.uint8)  # Normalize as uint8

    # Resize each image to (28, 28) using OpenCV
    resizedImages = []
    for image in imArray:
        resizedImage = cv2.resize(image.reshape(
            (28, 28)), (28, 28), interpolation=cv2.INTER_LINEAR)
        resizedImages.append(resizedImage)
    return resizedImages


def resizeMnistTo28x28Greyscale(imgArray):
    """
    Takes image array and resizes into 28x28 greyscale images
    
    Parameters:
    @imgArray: Array; contains numpy images 
    @Returns: resized array images 
    """
    img = Image.fromarray(np.uint8(imgArray*255), 'L')
    imgResized = img.resize((28, 28))
    imgResized = np.asarray(imgResized) / 255.
    return imgResized


def resizeMnistTo28x28RGB(imgArray):
    """Takes image array and resizes into 28x28 RBG images

    Parameters:
    @imgArray: Array containing images
    @Return: resized RGB array of images
    """
    img = Image.fromarray(np.uint8(imgArray*255), 'L')
    imgResized = img.resize((28, 28))
    # Convert to RGB and normalize
    imgRGB = np.asarray(imgResized.convert('RGB')) / 255.
    return imgRGB


def resizeImageTo28x28Greyscale(image_path):
    """Resizes any image to a 28x28 greyscale for conversion

    Parameters:
    @image_path: the filepath of image to be converted
    @Return: resized greyscale image
    """
    # Preprocess image
    img = Image.open(image_path).convert('RGB')
    imgResized = img.resize((28, 28))
    imgArray = np.asarray(imgResized) / 255.
    return imgArray


def resizeImageTo28x28RGB(image_path):
    # Preprocess image
    img = Image.open(image_path).convert('L')
    imgResized = img.resize((28, 28))
    imgArray = np.asarray(imgResized) / 255.
    return imgArray



def predictImage(image, model):
    """Predicts the image class with a selected model
    
    Parameters:
    @image: the image to be predicted
    @model: the model to predict with (LeNet, AlexNet, VGG)
    @Return: the predicted class of given image"""
    # initialise model
    tModel = model
    tModel.eval()  # set to evaluation mode

    # ADDED TO FIX PREDICTION- predicted image (on screen is stretched now tho)
    #------------------------------
    image = np.float32(image)

    # convert to grayscale if image has 3 channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgTensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = tModel(imgTensor)
        output = torch.softmax(output, dim=1)
        maxProb, predictedClass = torch.max(output, dim=1)
        predictions = predictedClass.item()
        return predictions, maxProb.item()


def labelToLetter(intLabel):
    """Converts Dataset label to relevant letter
    
    Parameters:
    @intLabel: int; label of an image
    @Returns: String; letter of the dataset label"""
    return chr(int(intLabel)+97).upper()
