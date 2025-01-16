# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

if __name__ == '__main__':
    # Data preprocessing and augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crop images with padding
        transforms.RandomHorizontalFlip(),    # Flip images horizontally
        transforms.ToTensor(),                # Convert to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize RGB
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Class labels
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Define the CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 32 filters
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 64 filters
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling
            self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Fully connected layer
            self.fc2 = nn.Linear(128, 10)  # Output layer
            self.dropout = nn.Dropout(0.5)  # Dropout for regularization

        def forward(self, x):
            x = torch.relu(self.conv1(x))  # Convolution + ReLU
            x = self.pool(x)  # Pooling
            x = torch.relu(self.conv2(x))  # Convolution + ReLU
            x = self.pool(x)  # Pooling
            x = x.view(-1, 64 * 8 * 8)  # Flatten
            x = torch.relu(self.fc1(x))  # Fully connected + ReLU
            x = self.dropout(x)  # Dropout
            x = self.fc2(x)  # Output layer
            return x

    # Instantiate and move the model to the device
    model = CNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Train the model
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loader_tqdm.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{num_epochs}]")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
