import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd

# Read the data
df = pd.read_csv('filtered_descriptions.csv')

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['Long Description'])

# Convert string labels to numerical format
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(df['CIS v8 Control Area'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels_encoded, test_size=0.2, random_state=42)

# Define neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Make predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calculate precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')
