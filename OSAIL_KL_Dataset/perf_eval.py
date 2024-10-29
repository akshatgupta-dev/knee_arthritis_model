from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch
from lib_and_paths import test_loader, class_names
from def_CNN import model

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion Matrix and Classification Report
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Evaluate the model on test data
evaluate_model(model, test_loader)
