import os
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(64 * 54 * 44, 128),
            nn.ReLU(),
            nn.Linear(128, 40),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.fc_stack(x)
        return x


model = NeuralNetwork()
print(model)

# Define the hyperparameters
batch_size =16 # The number of samples per batch
num_epochs = 3 # The number of times to iterate over the whole dataset
learning_rate = 0.01 # The learning rate for the optimizer

transform = transforms.Compose([
    transforms.ToTensor(), # Convert the images to tensors
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the pixel values with mean and std
])


image_path = '/root/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/img_align_celeba/img_align_celeba/'
attr = pd.read_csv('/root/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/list_attr_celeba.csv')
attr = attr.replace(-1, 0)
labels = attr.loc[:, attr.columns != 'image_id']

class CelebADataset(Dataset):
    def __init__(self, image_path, labels_df, transform=None):
        self.image_path = image_path
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        labels = self.labels_df.iloc[idx, 1:].values.astype('float32')

        img_path = os.path.join(self.image_path, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels


dataset = CelebADataset(image_path, attr, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test dataloader
for images, labels in dataloader:
    print(images.shape, labels)
    break

model = NeuralNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # The stochastic gradient descent optimizer

def accuracy(outputs, labels):
    # Konverter outputs til binære værdier (0 eller 1) baseret på en threshold
    preds = (outputs > 0.5).float()  # Threshold ved 0.5

    # Sammenlign hver output med den tilsvarende label
    correct = torch.sum(preds == labels).item()
    total = labels.numel()  # Det totale antal labels

    # Returnér accuracy som en procentdel
    return correct / total

# Define the training loop
def train(model, train_loader, criterion, optimizer, epoch):
    # Set the model to training mode
    model.train()
    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_acc = 0.0
    # Loop over the batches of data
    for i, (inputs, labels) in enumerate(train_loader):

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs) # Get the output logits from the model
        loss = criterion(outputs, labels) # Calculate the loss
        # Backward pass and optimize
        loss.backward() # Compute the gradients
        optimizer.step() # Update the parameters
        # Print the statistics
        running_loss += loss.item() # Accumulate the loss
        running_acc += accuracy(outputs, labels) # Accumulate the accuracy
        if (i + 1) % 200 == 0: # Print every 200 batches
            print(f'Epoch {epoch}, Batch {i + 1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
            running_loss = 0.0
            running_acc = 0.0

# Define the test loop
def test(model, test_loader, criterion):
    # Set the model to evaluation mode
    model.eval()
    # Initialize the loss and accuracy
    test_loss = 0.0
    test_acc = 0.0
    # Loop over the batches of data
    with torch.no_grad(): # No need to track the gradients
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs) # Get the output logits from the model
            loss = criterion(outputs, labels) # Calculate the loss
            # Print the statistics
            test_loss += loss.item() # Accumulate the loss
            test_acc += accuracy(outputs, labels) # Accumulate the accuracy
    # Print the average loss and accuracy
    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}')

for epoch in range(1, num_epochs + 1):
    train(model, dataloader, criterion, optimizer, epoch) # Train the model
    test(model, dataloader, criterion) # Test the model