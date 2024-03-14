import numpy as np
import torch
import torchvision
import os.path
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

BATCH_SIZE = 64

# Utils
def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


# define the transformations that will be performed on images
# transforms.ToTensor converts images to numbers and then to Tensors
# transforms.Normalize normalizes tensors
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])


# Download data and transform it
trainset = datasets.MNIST('data_train', download=True, train=True, transform=transform)
valset = datasets.MNIST('data_val', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)


# Exploratory analysis
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(images.shape)
print(labels.shape)

# Display the images
figure = plt.figure("Digits")
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(8, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')



# Build the neural network
# Input layer (784) -> Hidden layer (128) -> Hidden layer (64) -> Output layer (10)
INPUT_SIZE = 784
HIDDEN_SIZE = [128, 64]
OUTPUT_SIZE = 10
MODEL_PATH = './model.pt'

# check for saved model
if not os.path.isfile(MODEL_PATH):
    print("Model not found, creating new model...")

    model = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE[0]),
                        nn.ReLU(),
                        nn.Linear(HIDDEN_SIZE[0], HIDDEN_SIZE[1]),
                        nn.ReLU(),
                        nn.Linear(HIDDEN_SIZE[1], OUTPUT_SIZE),
                        nn.LogSoftmax(dim=1))
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)

    logps = model(images.cuda())
    loss = criterion(logps, labels.cuda())


    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images.cuda())
            loss = criterion(output, labels.cuda())
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    torch.save(model, MODEL_PATH) 
else:
    print("Loading model ...")
    model = torch.load(MODEL_PATH)
    print("Model loaded.")

# Evaluation
images_iter = iter(valloader)
for i in range(BATCH_SIZE):
    images, labels = next(images_iter)

    img = images[0].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img.cuda())

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.cpu().numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    view_classify(img.view(1, 28, 28), ps)
    plt.show()