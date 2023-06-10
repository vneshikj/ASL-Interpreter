# set the numpy seed for better reproducibility
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from ConvertDataset import ConvertDataset
np.random.seed(48)
# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import imutils
import torch
import cv2
from PIL import Image
from torchvision import transforms
from models import LeNet, AlexNet
from predict import resizeImageTo28x28RGB

class Predict(QObject):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def startPredictingCSV(self, testCsvFilePath, modelFilePath, SIZE):

        # check if input file is a csv or png
        def getAccuracy(correct, SIZE):
                    accuracy = float((correct/SIZE)*100)
                    return (accuracy)
        
        print("[INFO] loading the test dataset")
        testData = ConvertDataset(testCsvFilePath)
        idxs = np.random.choice(range(0,len(testData)), size =(SIZE,))
        testData = Subset(testData, idxs)

        testDataLoader = DataLoader(testData,batch_size=self.hyperparameters["BatchSize"])

        # load model and set it to evaluation mode 
        model = torch.load(modelFilePath).to(self.device)
        model.eval()
        correct = 0

        with torch.no_grad():
            for(image,label) in testDataLoader:
                origImage = image.numpy().squeeze(axis=(0,1))
                gtLabel = testData.dataset.classes[label.numpy()[0]]

                image = image.to(self.device)
                pred = model(image)
                
                # find class label with highest probability
                idx = pred.argmax(axis=1).cpu().numpy()[0]
                predLabel = testData.dataset.classes[idx]

                # convert image from grayscale to RGB so we can draw
                # on it and resize
                origImage = np.dstack([origImage]*3)
                origImage = imutils.resize(origImage,width=128)

                # draw predicted class label on it
                color = (0,255,0) if gtLabel == predLabel else (0,0,255)
                if(gtLabel == predLabel):
                    correct += 1
                cv2.putText(origImage, str(gtLabel), (2,25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
                
                # dictionary to convert labels to letters
                alphabet_dict = { 0: '-', 1: '-', 2: '-', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: '7', 8: 'h', 9: 'i', 10: '-',
                                    11: '?', 12: 'l', 13: '13', 14: '14', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: '-', 20: '20',
                                    21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y' 
                }

                # display result in terminal and show input image
                #print("[INFO] correct label: {}, predicted label: {}".format(alphabet_dict[gtLabel], alphabet_dict[predLabel]))
                print("[INFO] correct label: {}, predicted label: {}".format(gtLabel, predLabel))
                cv2.imshow("image", origImage)
                cv2.waitKey(0)
                return {"accuracy":"Accuracy: {:2f}%".format(getAccuracy(correct, SIZE)) }

            print("Accuracy: {:2f}%".format(getAccuracy(correct, SIZE)))

    
    def predictImageLeNet(self, imagePath, modelFilePath):
        model = LeNet(3,29)
        model.load_state_dict(torch.load(modelFilePath)["model_state_dict"])
        model.eval()

        inputData = resizeImageTo28x28RGB(imagePath)
        with torch.no_grad():
             output = model(inputData)
             predictions = torch.argmax(output, dim=1).item()
             print(predictions)

    def predictCSVLeNet(testCsvFilePath, modelFilePath, lenSubset):
        print("[INFO] Loading test dataset")
        testData = ConvertDataset(testCsvFilePath) 
        idxs = np.random.choice(range(0,len(testData)), size =(lenSubset,))
        testData = Subset(testData, idxs)
        testDataLoader = DataLoader(testData, batch_size=1)

        # Load the pre-trained model
        model = LeNet(3,29)
        model.load_state_dict(torch.load(modelFilePath)["model_state_dict"])
        model.eval()

        print("[INFO] predicting subset of csv")
        with torch.no_grad():
            for(image,label) in testDataLoader:
                output = model(image)
                predictions = torch.argmax(output,dim=1).item()
                print("label: {}, predicted: {}".format(label, predictions))

        