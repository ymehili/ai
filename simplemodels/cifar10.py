# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # Class labels
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Define the CNN architecture
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 10)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x

    # Instantiate and move the model to the device
    model = CNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Adam optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Train the model
    model.train()
    num_epochs = 30
    for epoch in range(num_epochs):
        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print batch progress
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Batch [{batch_idx+1}/{total_batches}], "
                  f"Loss: {loss.item():.4f}")
        scheduler.step()

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
