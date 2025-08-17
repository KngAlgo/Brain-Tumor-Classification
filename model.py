import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.callbacks import ModelCheckpoint

from keras.utils import image_dataset_from_directory

train_imgs = image_dataset_from_directory('/workspace/image_class_app/data', subset='training', validation_split=0.2, batch_size=64, seed=123, image_size=(256, 256))

val_imgs = image_dataset_from_directory('/workspace/image_class_app/data', subset='validation', batch_size=64, validation_split=0.2, seed=123, image_size=(256, 256))

# Apply data augmentation and optimization
train_imgs = train_imgs.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_imgs = val_imgs.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# /workspace/image_class_app/data

def build_model():
    input = keras.Input(shape=(256, 256, 3))
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(input)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Normalization
    x = layers.Rescaling(1./255)(x)

    # Deeper CNN architecture with better regularization
    filter_sizes = [32, 64, 128]
    for i, filters in enumerate(filter_sizes):
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.BatchNormalization()(x)
    
    # Global average pooling instead of flatten
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(3, activation='softmax')(x)
    
    return Model(input, output)

model = build_model()

# Better optimizer and learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Add callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
    keras.callbacks.ModelCheckpoint("brain_tumor_v2.h5", save_best_only=True, monitor='val_accuracy')  # use HDF5 format to avoid native Keras format options error
]

model.fit(
    train_imgs, 
    epochs=30, 
    validation_data=val_imgs, 
    callbacks=callbacks
)

# Save final model in HDF5 format
model.save('brain_tumor.h5')
