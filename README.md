# 🧠 Brain Tumor Classification

A deep learning application for classifying brain tumor types from MRI scans using TensorFlow/Keras with a web interface built in Flask.

## 📋 Table of Contents
- [Overview](#overview)
- [Model Versions](#model-versions)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Web Application](#web-application)
- [Performance](#performance)
- [Installation](#installation)

## 🔬 Overview

This project implements a Convolutional Neural Network (CNN) to classify brain tumors into three categories:
- **Glioma** - A type of tumor that starts in the glial cells
- **Meningioma** - A tumor that arises from the meninges 
- **Pituitary Tumor** - A growth in the pituitary gland

The model processes 256x256 MRI scan images and provides confidence scores for each tumor type.

## 🚀 Model Versions

### Evolution of Our Models

#### 1. `model.py` (Initial Version)
- Basic CNN architecture
- Simple data augmentation
- **Issues**: Stuck at 33.3% validation accuracy due to data split problems

#### 2. `simple_improved_model.py` 
- Enhanced CNN with better regularization
- Class weights for balanced training
- **Issues**: Still suffered from alphabetical data splitting

#### 3. `train_final_model.py`
- Implemented stratified data splitting
- Achieved 50.2% validation accuracy
- **Issues**: Still overfitting (77% train vs 50% val)

#### 4. `brain_tumor_v2.h5` (Current Best Model)
- **Optimized architecture** with proper regularization
- **Advanced data augmentation** with reduced intensity
- **Label smoothing** for better generalization
- **Learning rate scheduling** and early stopping
- **Performance**: Stable training with better validation accuracy

## 📊 Data Processing

### Dataset Structure
```
data/
├── glioma/          # 700 images
├── meningioma/      # 700 images
└── pituitary tumor/ # 700 images
```

### Data Manipulation Journey

1. **Initial Problem**: Imbalanced dataset (1418 glioma, 700 meningioma, 922 pituitary)
2. **Balancing**: Moved excess images to `extra_data/` folder to create 700 images per class
3. **Splitting Issue**: Used `image_dataset_from_directory` with `validation_split=0.2`
   - **Problem**: Alphabetical sorting put all pituitary tumors in validation set
   - **Solution**: Added `shuffle=True` to `image_dataset_from_directory`

4. **Final Split**:
   - Training: 560 images per class (1,680 total)
   - Validation: 140 images per class (420 total)

### Data Augmentation Pipeline
```python
# Applied in this order:
1. RandomFlip("horizontal")     # Mirror images horizontally
2. RandomRotation(0.05)         # Small rotations (±2.9°)
3. RandomZoom(0.05)            # Slight zoom variations
4. GaussianNoise(0.05)         # Add subtle noise for robustness
5. Rescaling(1./255)           # Normalize pixel values to [0,1]
```

## 🏗️ Model Architecture

### Current Model (`brain_tumor_v2.h5`)

```python
Input (256, 256, 3)
    ↓
Data Augmentation Layers
    ↓
Rescaling (Normalization)
    ↓
3x Conv Blocks:
├── Conv2D(32/64/128, 3x3) → Conv2D → MaxPool2D → BatchNorm → Dropout(0.3)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu) → Dropout(0.5)
    ↓
Dense(3, softmax) # 3 classes
```

### Key Features
- **Global Average Pooling**: Reduces overfitting compared to Flatten
- **High Dropout Rates**: 0.3 in conv blocks, 0.5 before final layer
- **Batch Normalization**: Stabilizes training
- **Label Smoothing**: Reduces overconfidence (smoothing=0.1)

### Training Configuration
```python
Optimizer: Adam(lr=0.0001)          # Conservative learning rate
Loss: CategoricalCrossentropy(label_smoothing=0.1)
Callbacks:
├── ModelCheckpoint: Save best model
├── ReduceLROnPlateau: Dynamic learning rate
└── EarlyStopping: Prevent overfitting
```

## 💻 Usage

### 1. Training a New Model
```python
python model.py
```
This will:
- Load and preprocess the dataset
- Train the model with data augmentation
- Save the best model as `brain_tumor_v2.h5`

### 2. Using Pre-trained Model
```python
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_v2.h5')

# Preprocess your image
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction
predictions = model.predict(preprocess_image('your_image.png'))
class_names = ['glioma', 'meningioma', 'pituitary tumor']
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)
```

## 📈 Performance

### Why We Chose `brain_tumor_v2.h5`

1. **Solved Data Leakage**: Proper stratified splitting
2. **Reduced Overfitting**: 
   - Lower learning rate (0.0001)
   - Higher dropout rates
   - Early stopping
3. **Better Generalization**:
   - Label smoothing
   - Reduced augmentation intensity
   - Learning rate scheduling
4. **Stable Training**: Consistent validation metrics

### Training History
- **Previous Models**: 33.3% → 50.2% validation accuracy
- **Current Model**: Stable training with reduced overfitting
- **Key Improvement**: Fixed alphabetical data splitting issue

## 🛠️ Installation

### Requirements
```bash
pip install tensorflow==2.13.1
pip install flask
pip install pillow
pip install numpy
pip install matplotlib
pip install pandas
```

### Setup
1. Clone the repository
2. Install dependencies
3. Ensure your data is in the `data/` folder with proper structure
4. Run `python model.py` to train or `python app.py` to use the web interface

## 📁 Project Structure
```
├── app.py                 # Flask web application
├── model.py              # Current training script
├── brain_tumor_v2.h5     # Best trained model
├── data/                 # Training dataset
├── templates/            # HTML templates
├── static/              # CSS, JS, and assets
├── uploads/             # Temporary upload folder
└── README.md            # This file
```
---

**Model Performance**: This model represents the culmination of multiple iterations, data preprocessing improvements, and architectural optimizations to achieve reliable brain tumor classification from MRI scans.
