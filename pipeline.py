import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os

# Path to the labeled dataset
data_dir = r"persistent/AI_THEME_III/Knee_arthritis/OSAIL_KL_Dataset/Labeled"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset with ImageFolder
dataset = ImageFolder(root=data_dir, transform=transform)

# Split dataset into train, validation, and test (e.g., 70% train, 15% val, 15% test)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Confirm data loading
class_names = dataset.classes
print(f"Classes: {class_names}")
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

import torch.nn as nn
import torch.nn.functional as F

class KneeXRayCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(KneeXRayCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = KneeXRayCNN(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003 )


from tqdm import tqdm
import matplotlib.pyplot as plt

# Modified training function with progress bar
def train_model_with_progress(model, criterion, optimizer, train_loader, val_loader, num_epochs=20):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Use tqdm to show progress
        for images, labels in tqdm(train_loader, desc="Training Batches"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
        
        train_accuracy = correct_train / len(train_set)
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        
        # Validation with tqdm
        model.eval()
        correct_val = 0
        val_loss = 0.0
        for images, labels in tqdm(val_loader, desc="Validation Batches"):
            with torch.no_grad():
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
        
        val_accuracy = correct_val / len(val_set)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        
        print(f"Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Train the model with progress
train_losses, val_losses, train_accuracies, val_accuracies = train_model_with_progress(
    model, criterion, optimizer, train_loader, val_loader, num_epochs=20
)

# Plot Loss and Accuracy Curves
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
