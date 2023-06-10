# views
from ui.views.landingview.landingview import LandingView
from ui.views.trainview.trainview import TrainView
from ui.views.loadingview.loadingview import LoadingView
from ui.views.testingview.testingview import TestingView
# controllers
from ui.controllers.loadingcontroller import LoadController
from ui.controllers.traincontroller import TrainController
from ui.controllers.testingcontroller import TestingController
# models
from ui.models.trainmodel import TrainModel
from ui.models.loadingmodel import LoadModel
from ui.models.testingmodel import TestingModel

class MyController:

    # init all views, controllers and models
    def __init__(self, layout):
        self.layout = layout
        # Train
        self.trainView = TrainView(self)
        self.trainModel = TrainModel()
        self.trainController = TrainController(self.trainModel, self.trainView)
        # landing
        self.landingView = LandingView(self)
        # Load
        self.loadingView = LoadingView(self)
        self.loadModel = LoadModel()
        self.loadController = LoadController(self.loadModel, self.loadingView)
        # Test
        self.testingView = TestingView(self)
        self.testingModel = TestingModel(self.loadModel)
        self.testingController = TestingController(self.testingModel, self.testingView)
        print("App Loaded!")
        
    def showTrainView(self):
        self.removeCurrentView()
        self.layout.addWidget(self.trainView)

    def showLoadingView(self):
        self.removeCurrentView()
        self.layout.addWidget(self.loadingView)

    def showLandingView(self):
        self.removeCurrentView()
        self.layout.addWidget(self.landingView)

    def showTestingView(self):
        self.removeCurrentView()
        self.layout.addWidget(self.testingView)

    def removeCurrentView(self):
        [self.layout.itemAt(i).widget().setParent(None)
         for i in reversed(range(self.layout.count()))]
