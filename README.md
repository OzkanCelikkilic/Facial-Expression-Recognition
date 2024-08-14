# Facial Expression Recognition with CNN

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for recognizing facial expressions. The model is trained on the FER2013 dataset and can classify images into seven distinct categories: angry, disgust, fear, happy, neutral, sad, and surprise.

## Project Overview

Facial expression recognition is a crucial aspect of human-computer interaction, security systems, and psychological research. This project leverages deep learning techniques to classify facial expressions from grayscale images.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results and Evaluation](#results-and-evaluation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [References](#references)

## Dataset

The dataset used in this project is FER2013 dataset, consisting of grayscale 48*48 images categorized into seven classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Due to size constraints, only a small subset of the dataset is included in this repository for demonstration purposes. The full dataset can be accessed [here](https://www.kaggle.com/datasets/msambare/fer2013).

## Model Architecture

The CNN model used in this project has the following architecture:

- **Conv2D Layers**: Extract features from the input images.
- **MaxPooling2D Layers**: Downsample the feature maps.
- **Dropout Layers**: Prevent overfitting by randomly setting a fraction of input units to 0.
- **Flatten Layer**: Convert the 2D matrices into a 1D vector.
- **Dense Layers**: Perform classification based on the extracted features.
- **Softmax Output**: Classifies the image into one of the seven categories.

## Training Process

The model was trained for 20 epochs with a batch size of 32. Early stopping was applied to prevent overfitting.

The model was compiled with the following settings:

- **Optimizer**: Adam with a learning rate of 0.0001
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## Results and Evaluation

### Accuracy and Loss Graphs

The following graphs depict the model's performance during training:

- **Accuracy**: Shows the training and validation accuracy over the epochs.
- **Loss**: Shows the training and validation loss over the epochs.

![20epoch](https://github.com/user-attachments/assets/96a016a0-457a-42c7-b465-eea72f856f11)


### Confusion Matrix

The confusion matrix heatmap visualizes the performance of the model on the test set, indicating the correctly and incorrectly predicted classes.


![confusion](https://github.com/user-attachments/assets/9e403851-210e-4005-aae0-190c461696a9)

### Class Distribution

The bar graph below shows the distribution of classes in the training dataset.

![Figure 2024-05-17 180225](https://github.com/user-attachments/assets/aa9290b3-c15b-4b7a-9f37-c32f09fec39d)


### Some Screenshots I took when I used webcam
![Ekran görüntüsü 2024-05-18 201801](https://github.com/user-attachments/assets/f165de40-599c-43d8-a187-51ed7a4dd121)

![Ekran görüntüsü 2024-05-18 202143](https://github.com/user-attachments/assets/36c63e82-578f-44d1-af01-0730ab93a7d5)

### Schematic Visualization of the Model

The model's layer-by-layer visualization is shown below, including the input and output shapes for each layer

![layers](https://github.com/user-attachments/assets/162100c5-8a8e-4599-8c9c-b92d8a2cde66)
![layers1](https://github.com/user-attachments/assets/8e8dbaa5-b63b-42f4-bef4-8d85681977f4)
![layers2](https://github.com/user-attachments/assets/ec949d64-4966-486d-854b-a8b769355143)


## Usage

To run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/OzkanCelikkilic/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition
pip install -r requirements.txt




