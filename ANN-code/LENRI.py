import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


#%%
# Data Preparation

# Load CSV data
data = pd.read_csv("Data/all_features.csv")

# Extract features and labels
def extract_species(name):
    """Extract species label from event name."""
    return 0 if "C" in name else 1

data['species'] = data['Name'].apply(extract_species)
X = data.iloc[:, 1:5].values  # Select columns with feature data (assuming columns 1-4 are features)
y = data['species'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to categorical (one-hot encoding for binary classification)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#%%

# Define the LENRI model architecture
LENRI = Sequential([
    Dense(32, input_shape=(4,), activation='relu'),  # Input layer with 4 features
    Dropout(0.2),                                    # Dropout for regularization
    Dense(16, activation='relu'),                    # Hidden layer
    Dropout(0.2),                                    # Dropout for regularization
    Dense(8, activation='relu'),                     # Another hidden layer
    Dense(2, activation='softmax')                   # Output layer for binary classification
])

# Compile LENRI
LENRI.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# LENRI
LENRI.summary()

#%%

# Train the model
history = LENRI.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

#%%

# LENRI Evaluation
test_loss, test_accuracy = LENRI.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
