import time
from lib_and_paths import torch
from lib_and_paths import train_loader, val_loader, train_set, val_set
from def_CNN import model, criterion, optimizer

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
        
        train_accuracy = correct_train / len(train_set)
        
        # Validation
        model.eval()
        correct_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
        
        val_accuracy = correct_val / len(val_set)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)
