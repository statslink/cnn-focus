# CNN Focus: Learning-Based Microscopic Autofocus

## Overview
This project explores microscopic autofocus as a machine learning problem. Instead of relying purely on classical sharpness-based methods, convolutional neural networks (CNNs) are used to identify the in-focus image within a 3D microscope image stack.

Two approaches are considered:
- Classification: predicting the number of steps to reach focus
- Regression: predicting a continuous sharpness value

## Problem Setting
An image stack is defined as:

S = {S_0, S_1, ..., S_m}

The goal is to identify the in-focus slice efficiently.

## Approaches

### Classification
Images are labelled by distance from the focused slice.
Class 0 represents acceptable focus, higher classes indicate distance.

Prediction rule:
prediction = sum(sigmoid(logits) > 0.5)

### Regression
Each image is assigned a sharpness value:
s(x) -> R

The model learns to approximate this function.

## Data
- 13 image stacks
- Magnifications: 10x, 20x, 40x, 60x, 100x
- Grayscale images

Preprocessing:
- Resize to 300x300
- Normalize
- Convert to tensors

## Model Architecture
- 3 convolutional layers (32, 64, 96 channels)
- Max pooling + ReLU
- Adaptive average pooling
- Fully connected layers
- Dropout (p = 0.1)

## Training
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16

## Results
- 70–80% accuracy on single magnification datasets
- ~30% on mixed datasets
- Strong regression performance

## Limitations
- Small dataset
- Limited generalisation across magnifications

## Future Work
- Hybrid models (classification + regression)
- Larger datasets
- Real-world deployment

## Technologies
Python, PyTorch, NumPy, scikit-learn, matplotlib

## Summary
CNNs can learn focus structure in microscopic images and offer a promising direction for intelligent autofocus systems.
