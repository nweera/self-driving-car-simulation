# Self-Driving Car Steering Angle Prediction

## Project Overview
This project focuses on predicting the steering angle of a car using images from a front-facing camera. It is an essential component of self-driving systems, aiming to improve autonomous driving accuracy in varied and realistic driving conditions.  

We developed a deep learning model using Convolutional Neural Networks (CNNs) and applied data balancing and augmentation techniques to simulate real-life driving scenarios such as turns, lighting changes, and road variations.  

---

## Features
- Load and preprocess driving datasets
- Data balancing to reduce bias towards straight-driving samples
- Image augmentation including zoom, pan, brightness adjustments, and flipping
- CNN-based steering angle prediction using Keras and TensorFlow
- Evaluation through training/validation loss and visual comparisons of predicted vs actual steering angles

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Project Context](#project-context)  
   - Problem Statement  
   - Objective  
   - Methodology  
3. [Tools and Technology Stack](#tools-and-technology-stack)  
4. [System Design and Architecture](#system-design-and-architecture)  
5. [Implementation](#implementation)  
   - Data Processing  
   - Data Balancing  
   - Data Augmentation  
   - Model Architecture  
6. [Evaluation and Results](#evaluation-and-results)  
7. [Conclusion and Future Work](#conclusion-and-future-work)  
8. [References](#references)  

---

## Introduction
Self-driving technology is rapidly evolving and aims to transform transportation. A key challenge is predicting steering angles accurately from camera input. This project tackles this challenge using deep learning to create a model that performs well even in non-ideal driving conditions.

---

## Project Context

### Problem Statement
Most driving datasets contain an overrepresentation of straight-driving samples, causing bias and poor performance in curves and turns.  

### Objective
Build a model capable of accurately predicting steering angles across diverse driving scenarios by:
- Cleaning and balancing the dataset
- Applying image augmentations to simulate realistic conditions

### Methodology
An agile approach was followed with four main sprints:

**Sprint 1:** Data loading and processing  
**Sprint 2:** Data balancing, train-validation split, augmentation, baseline CNN training  
**Sprint 3:** Image preprocessing, batch generator, Nvidia model implementation  
**Sprint 4:** Final training, environment setup, testing

---

## Tools and Technology Stack
- **Languages & Frameworks:** Python, TensorFlow, Keras  
- **Libraries:** Pandas, NumPy, Matplotlib, OpenCV, Albumentations  
- **Platform:** Google Colab  
- **Version Control:** Git  

---

## System Design and Architecture
- **Use Case Diagram:** Illustrates how users interact with the system  
- **Sequence Diagram:** Shows data flow from image input to steering prediction  
- **Class Diagram:** Defines the structure of modules and their relationships  

---

## Implementation

### Data Processing
- Loaded CSV logs with image paths and steering angles  
- Cleaned and normalized paths  
- Visualized steering angle distribution  

### Data Balancing
- Performed histogram analysis of angle frequency  
- Undersampled straight-driving images to reduce bias  

### Data Augmentation
Applied transformations to mimic real-life conditions:  
- **Zooming:** Simulate approaching objects  
- **Panning:** Shift images horizontally/vertically  
- **Brightness Adjustments:** Different lighting scenarios  
- **Flipping:** Horizontal flips with inverted steering angles  
- **Random Combinations:** Applied randomly for variety  

### Model Architecture
- CNN with layers: Conv2D, MaxPooling, Dropout, Flatten, Dense  
- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam  

---

## Evaluation and Results
- Monitored training and validation loss  
- Plotted predicted vs actual steering angles  
- Observed improvements in curves and varied lighting conditions  
- Addressed overfitting through augmentation and preprocessing  

---

## Conclusion and Future Work
We successfully trained a CNN to predict steering angles reliably across multiple driving scenarios. Future work may include:
- Integrating additional sensor data (e.g., LiDAR)  
- Testing on real-world autonomous vehicles  
- Exploring more advanced architectures for higher accuracy  

---

## References
- [YouTube Channel: deDSwithBappy]([https://youtube.com/playlist?list=PLkz_y24mlSJawbZzfJrrxZi0QNmJ5aP6&si=pqcxVFW0OXLOouoN](https://www.youtube.com/playlist?list=PLkz_y24mlSJawbZz-fJrrxZi0QNmJ5aP6))
