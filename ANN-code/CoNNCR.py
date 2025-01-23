"""
CoNNCR

Convolutional
Neural
Network
(for)
Classification
(of)
Recoils

"""

# %%

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam

# %%

# Import data for training

# X is list of image arrays
# y are 0 or 1 (carbon or fluorine)


# %%

# Split into 70% train, 15% validation, 15% test
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - train_ratio, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test,
    y_test,
    test_size=test_ratio / (test_ratio + validation_ratio),
    random_state=42,
)

# %%

# Define CoNNCR

input_shape = (768, 768, 1)  # Example shape, you can change it based on your data

# Define the model
model = Sequential()

# Conv Block 1
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(768, 768, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv Block 2
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv Block 3
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(2, activation="softmax"))

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Summary of the model
model.summary()


# %%

# Assuming X_train, y_train, X_val, y_val are preprocessed and ready
# where X_train and X_val are the image datasets and y_train, y_val are the labels

# Training the model
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_val, y_val),
    verbose=1,
)

# Evaluating the model
test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")
