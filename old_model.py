import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory

train_imgs = image_dataset_from_directory(
    '/workspace/image_class_app/data',
    subset='training',
    validation_split=0.2,
    batch_size=64,
    seed=123,
    image_size=(256, 256),
    shuffle=True,
    label_mode='categorical'  # one-hot labels for label smoothing
)

val_imgs = image_dataset_from_directory(
    '/workspace/image_class_app/data',
    subset='validation',
    validation_split=0.2,
    batch_size=64,
    seed=123,
    image_size=(256, 256),
    shuffle=True,
    label_mode='categorical'
)

# Apply data augmentation and optimization
train_imgs = train_imgs.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_imgs = val_imgs.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# /workspace/image_class_app/data

def build_model():
    input = keras.Input(shape=(256, 256, 3))
    
    # Normalization first
    x = layers.Rescaling(1./255)(input)
    
    # Data augmentation (only during training)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)  # Reduced rotation
    x = layers.RandomZoom(0.05)(x)     # Reduced zoom
    x = layers.GaussianNoise(0.05)(x)  # Reduced noise

    # Deeper CNN architecture with better regularization
    filter_sizes = [32, 64, 128]
    for filters in filter_sizes:
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)  # Increased dropout
    
    # Global average pooling instead of flatten
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Higher dropout before final layer
    output = layers.Dense(3, activation='softmax')(x)
    
    return Model(input, output)

model = build_model()

# Better optimizer and learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Much lower learning rate
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # use categorical for smoothing
    metrics=['accuracy']
)

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("brain_tumor_v2.h5", save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
]

model.fit(
    train_imgs, 
    epochs=25,  # Reduced epochs 
    validation_data=val_imgs, 
    callbacks=callbacks,
    verbose=1
)
