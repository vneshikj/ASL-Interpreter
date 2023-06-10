# Welcome to Goats ML Repository
## Introduction

>COMPSYS 302 Project 1 repository by "The Goats" (Team 9) 
### Team Members:
- Benson 
- Viktor
- Ian

## Getting Started
### Prerequisites
- [Python version ***3.9.2***](https://www.python.org/downloads/release/python-392/) *OR* [Conda](https://www.anaconda.com/download/)
- [Pip](https://pip.pypa.io/en/stable/installation/) as a package manager


*Note that it is recommended to use a virtual environment such as conda to prevent conflicts*
### Conda setup (Optional)

Run from the root directory of this repo (the one that contains this *README.md*):
```
conda create --name goatsml python=3.9.2
```
Type ```y``` when prompted
```
conda activate goatsml
```
Move on to the next section and follow the instructions if this is done
### Running the application

Open a terminal in the root directory of this repo (the one that contains this *README.md*)
Run
```
python INSTALL.py
```
To make sure that the packages required for this project are installed

Afterwards, from the same directory run 

```
python main.py
```

## Inside the application
### Available models
As per the project requirements, there are three pre-defined models that we have implemented for training

- LeNet
- AlexNet
- VGG

### Training the models
With Goats ML, the user is able to train the model through using the MNIST dataset in a csv format (can be found [*here*](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and use the test dataset) or using a folder-based dataset that follows the following structure:
- Has 26 distinct folders each representing an alphabet letter
- Each folder is labelled a letter of the English alphabet (A-Z)

The dataset can be split into its train test split by making use of the slider on the *dataset* screen.

After the datset is loaded, the following hyperparameters can be updated from the *hyperparameters* tab:
- Epochs
- Batch size

The default model architecture is LeNet, however a different option can be selected from the *models* tab.

To start the training, in the *train* tab click the *start* button and wait for the progress to reach 100%.


### Saving the model after training
When the model is done training (represented by the progress bar being at 100%), clicking the *save* button will open a dialog to save the trained model at the specified file location. The model will be saved with the following information 

- Name
- Architecture
- Train/Validation ratio
- Epochs
- Batch size

These are all visible when the models are loaded.

### Making predictions
Note that to start the predictions the following prerequiesites are required:
- A trained model in pth format (*MUST* be a model that was trained and saved using Goats ML)
- Training images in the format of:
    - CSV of MNIST format (we use [this](https://www.kaggle.com/datasets/datamunge/sign-language-mnist))
    - Individual images of any valid format
    - A working webcam

To start the training:
1. Upload a model through the *load* screen
2. Go to the *test* screen and choose one of the image options
*Note you need to start the webcam by clicking the button to predict that way*

### Loading Models
To load a model, the make sure you have the following prerequiesites:
- A trained model from using Goats ML in a *.pth format

Once you have all of the above:
1. Go to the *load* screen and click on *upload model* 
2. Afterwards you can click on the model and view information such as the *epochs*, *train validation ratio*, and *batch size*

### Dataset Viewer
Goats ML features a dataset viewer that can be accessed from the *train* screen. This screen allows you to upload a MNIST format 28x28 dataset or a folder dataset that complies with the standards outlined earlier. 

Once the dataset is loaded you may:
- View all the signs
- Filter out signs
- View the number of each sign

## Screenshots
![image](https://user-images.githubusercontent.com/100653148/233604251-73de5e46-6245-4083-bd30-fa598a452dce.png)
![image](https://user-images.githubusercontent.com/100653148/233604349-1f36fd2c-e873-4e8e-8b38-dc666d208c1b.png)
![image](https://user-images.githubusercontent.com/100653148/233604483-fe88e098-daf5-4c9b-8e31-12db85941a2e.png)
![image](https://user-images.githubusercontent.com/100653148/233604597-36138ebe-41f2-4269-acd4-b34963097c12.png)



