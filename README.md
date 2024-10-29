# Knee Osteoarthritis Classification Using CNNs

This repository contains my work on building a Convolutional Neural Network (CNN) model to classify the severity of knee osteoarthritis from X-ray images. The goal of this project was to apply deep learning techniques to analyze and score knee osteoarthritis severity based on the Kellgren-Lawrence (KL) grading system, using the [OSAIL Knee Osteoarthritis KL Scoring Dataset](https://www.kaggle.com/datasets/peymannejat/osail-knee-osteoarthritis-kl-scoring-dataset).

## Project Overview

The dataset contains labeled X-ray images, structured into five categories (0 to 4), each representing a level of osteoarthritis severity:
- **0**: No osteoarthritis
- **4**: Severe osteoarthritis

### Key Steps I Followed

1. **Data Preparation**:
   - The dataset is structured with images in labeled folders from `0` to `4`, representing the KL grades.
   - I split the data into training, validation, and test sets to ensure balanced evaluation metrics across all classes.

2. **Model Architecture**:
   - I built a Convolutional Neural Network (CNN) architecture using PyTorch, designed to capture the visual patterns in the X-ray images associated with different KL grades.
   - The model uses a series of convolutional, pooling, and fully connected layers with ReLU activations and dropout for regularization.

3. **Training Process**:
   - I trained the model with the cross-entropy loss function and used an Adam optimizer to adjust the learning rate dynamically.
   - The training was conducted over multiple epochs with real-time progress tracking using `tqdm`, which allowed me to monitor each batch's progress within every epoch.

4. **Evaluation and Metrics**:
   - To assess the model’s performance, I calculated the training and validation accuracy and loss for each epoch.
   - I used plots to visualize the model's performance over epochs, specifically tracking both training and validation loss and accuracy.

### Results

After training, I obtained the following results:
- Training and validation loss decreased consistently over epochs, indicating good convergence.
- The model showed significant accuracy across the KL grades, with the final training and validation accuracy being as shown in the screenshot below.

![Training and Validation Results](image.png)

> Screenshot shows the final results from my training, including loss and accuracy curves.

### Insights and Learnings

This project helped me gain experience with CNNs and PyTorch for medical imaging tasks. Some key takeaways include:
- Proper data splitting ensures unbiased evaluation, especially in medical image analysis.
- Regularization techniques, like dropout, play an important role in reducing overfitting on small datasets.
- Visualizing training progress and performance metrics is essential for debugging and understanding model behavior.

## Future Work

Further work could explore:
- Data augmentation to increase the dataset's variability and improve generalization.
- Hyperparameter tuning, particularly with respect to learning rate and layer configurations, to enhance performance.
