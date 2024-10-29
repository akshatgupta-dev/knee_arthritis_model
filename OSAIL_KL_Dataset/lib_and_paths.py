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
