import numpy as np
import torch
import torchvision
import os.path
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from utils import view_classify

class Model:

    def __init__(self, input_size, hidden_size, output_size, trainloader, model_path = './model.pt') -> None:
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model_path = model_path

        if not os.path.isfile(model_path):
            print("Model not yet created, creating new model...")
            self.__create_model(trainloader)
        else:
            print("Loading model ...")
            self.model = torch.load(model_path)
            print("Model loaded.")

    def __create_model(self, trainloader):
        self.model = nn.Sequential(nn.Linear(self.input_size, self.hidden_size[0]),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size[1], self.output_size),
                        nn.LogSoftmax(dim=1))
        print(self.model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.model.to(device)

        criterion = nn.NLLLoss()
        images, labels = next(iter(trainloader))
        images = images.view(images.shape[0], -1)

        logps = self.model(images.cuda())
        loss = criterion(logps, labels.cuda())

        # Training
        optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)
        time0 = time()
        epochs = 15
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
            
                # Training pass
                optimizer.zero_grad()
                
                output = self.model(images.cuda())
                loss = criterion(output, labels.cuda())
                
                #This is where the model learns by backpropagating
                loss.backward()
                
                #And optimizes its weights here
                optimizer.step()
                
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)

        torch.save(self.model, self.model_path) 

    def eval_single(self, img):
        print(img)
        print(img.shape)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = self.model(img.cuda())

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        print("Predicted Digit =", probab.index(max(probab)))
        view_classify(img.view(1, 28, 28), ps)
        plt.show()

    def evaluate(self, valloader, batch_size):
        images_iter = iter(valloader)
        for i in range(batch_size):
            images, labels = next(images_iter)

            img = images[0].view(1, 784)
            self.eval_single(img)