import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm  # For progress bars
import numpy as np

# --- 1. Data Loading and Preprocessing ---

# Define transformations
if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize to [-1, 1] (MNIST-specific values)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)


    # Split train_dataset into train and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True) # Larger batch size for testing

    # --- 2. Model Definition (High-Accuracy CNN) ---

    class HighAccuracyMNISTNet(nn.Module):
        def __init__(self):
            super(HighAccuracyMNISTNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 512),  # Adjusted for the output size of the convolutional layers
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)  # 10 output classes (digits 0-9)
            )
            # Apply Kaiming initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)


        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # --- 3. Training Setup ---

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    model = HighAccuracyMNISTNet().to(device)

    criterion = nn.CrossEntropyLoss()  # Standard loss for multi-class classification
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # AdamW with weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True) # Reduce LR on plateau

    # --- 4. Training Loop ---

    def train(model, device, train_loader, optimizer, criterion, epoch):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc

    def validate(model, device, val_loader, criterion):
        model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # No gradient calculation during validation
            for data, target in tqdm(val_loader, desc="[Validate]"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc

    def test(model, device, test_loader):
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # No gradient calculation during testing
            for data, target in tqdm(test_loader, desc="[Test]"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_acc = 100. * correct / total
        return test_acc


    num_epochs = 50
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5 # Early Stopping patience.

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = validate(model, device, val_loader, criterion)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning Rate Scheduling
        scheduler.step(val_loss)  # Pass the validation loss to the scheduler

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_mnist_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break


    # Load the best model
    model.load_state_dict(torch.load('best_mnist_model.pth'))

    test_acc = test(model, device, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")