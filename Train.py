# ECE452 Project 1, Part 1
# Group 8 Members: Arnav R., Raghav B., and Daniel A.

# Inspiration comes from kaggle.com/code/geekysaint/solving-mnist-using-pytorch 
import re
import os
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader

#----------------------------
# Class for the neural network
#----------------------------
class MnistCnn(nn.Module):
    # Two convolution and pooling layers, 3x3 with 1 cell padding each, and
    # max pooling also 2x2.
    def __init__(self):
        super(MnistCnn, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.FullyConnectedLayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    # Forward pass through convolutional and fully connected layers
    def forward(self, x):
        x = self.ConvLayers(x)
        x = self.FullyConnectedLayers(x)
        return x

    # Training function. Uses Mean Squared Error loss with Stochastic
    # Gradient Descent optimizer.
    def fit(self, images, labels, epochs: int = 5, lr: float = 0.001, batch: int = 64, plot_name=None):
        images = torch.stack(images)
        labels = torch.tensor(labels)
        dataset = TensorDataset(images, labels)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        device = torch.device("cpu")
        self.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        labels_OneHot = F.one_hot(labels, num_classes=10).float()

        # just puts the model in training mode, doesn't train it
        self.train()

        LossHistory = []
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels_OneHot)
                loss.backward()
                optimizer.step

                total_loss += loss.item()

            avg_loss = total_loss/len(train_loader)
            LossHistory.append(avg_loss)
            print(f'Epoch [{epoch+1}/{epochs}] complete. Avg loss: {avg_loss:.4f}')

        if plot_name:
            self.PlotLoss(LossHistory, plot_name)

    def PlotLoss(self, LossHistory, plot_name):
        if not plot_name.endswith('.png'):
            plot_name += '.png'

        plt.figure(figsize=(10,5))
        plt.plot(LossHistory, color='red', label='Training Loss')
        plt.title('Training Performance')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(PlotName)
        plt.close()
        print(f'Loss plot saved as {PlotName}')

#----------------------------
# Convert all training images to tensors 
#----------------------------
def ConvImageToTensor(RootFilePath: str) -> tuple[list[int], list[torch.FloatTensor]]:
    try:
        # Convert to Pillow image, and then use ToTensor function
        filenames = [f for f in os.listdir(RootFilePath) if f.endswith('.tif')]
        images = []
        labels = []
        for file in filenames:
            t = Image.open(os.path.join(RootFilePath, file))
            images.append(ToTensor()(t))
            labels.append(int(re.search(r'_(\d)_', file)[1]))

        return labels, images
    except FileNotFoundError:
        print(f'Specified root file path {RootFilePath} doesn\'t exist or smth')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train MnistCnn on MNIST training dataset")

    parser.add_argument('--dir', type=str, default='./Provided/Part1/test_data/', help='Path to TIF images.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--batch', type=int, default=64, help='Batch size for training')
    parser.add_argument('--output', type=str, default='model.pt', help='Name of output pt file')
    parser.add_argument('--plot', nargs='?', const='loss_plot.png', default=None, help='Enable plotting of loss curves.')

    args = parser.parse_args()

    labels, images = ConvImageToTensor(args.dir)

    model = MnistCnn()
    model.fit(images, labels,
              epochs=args.epochs,
              lr=args.lr,
              batch=args.batch,
              plot_name=args.plot)

    finalname = args.output if args.output.endswith('.pt') else args.output + '.pt'
    torch.save(model.state_dict(), finalname)
