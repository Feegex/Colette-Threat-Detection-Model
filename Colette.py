# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:34:16 2024

@author: nigel
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import time
import shap
import pandas as pd

# 1. Custom Dataset Handling - Use Real Dataset or Synthetic Data
def load_dataset(real_data_path=None, n_samples=1000, n_features=20):
    if real_data_path:
        # Load real dataset from CSV or other format
        data = pd.read_csv(real_data_path)
        features = data.drop(columns=['label'])  # Assumes 'label' column has target labels
        labels = data['label']
        return features, labels
    else:
        # Synthetic data generation if no real data path is provided
        from sklearn.datasets import make_classification
        features, target_labels = make_classification(n_samples=n_samples, n_features=n_features, n_informative=15, n_classes=2)
        return features, target_labels

# Preprocessing the dataset
def preprocess_data(features, target_labels, test_size=0.3):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target_labels, test_size=test_size)
    return X_train, X_test, y_train, y_test

# Custom Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.feed_forward_network = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embedding_dim)]
        )
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layer_norm1(inputs + attention_output)
        ffn_output = self.feed_forward_network(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)

# Build model
def build_transformer_model(input_shape, num_heads=2, key_dim=64, ff_dim=128, learning_rate=1e-4):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(key_dim)(inputs)
    x = layers.Reshape((1, key_dim))(x)
    transformer_block = TransformerBlock(embedding_dim=key_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(model, training_features, training_labels, epochs=30, batch_size=32, validation_split=0.2):
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    history = model.fit(training_features, training_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[lr_scheduler, early_stopping])
    return history

# Evaluate model
def evaluate_model(model, test_features, test_labels):
    loss, accuracy = model.evaluate(test_features, test_labels)
    print(f"Test Accuracy: {accuracy:.2f}")
    return accuracy

# Real-Time Monitoring Simulation
def real_time_monitoring(model, incoming_data_stream):
    print("Starting real-time threat monitoring...")
    for data in incoming_data_stream:
        prediction = model.predict(data.reshape(1, -1))  # Reshape for single data input
        print(f"Real-Time Prediction: {prediction[0][0]:.2f}")
        time.sleep(2)  # Simulating delay between incoming data

# Interpretability with SHAP
def explain_model_shap(model, training_features, test_features):
    explainer = shap.KernelExplainer(model.predict, training_features)
    shap_values = explainer.shap_values(test_features)
    shap.summary_plot(shap_values, test_features)

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load dataset (customizable)
    features, labels = load_dataset(real_data_path=None)  # Use real_data_path="path/to/real_dataset.csv" if available
    training_features, test_features, training_labels, test_labels = preprocess_data(features, labels)

    # Build and train model
    colette_transformer_model = build_transformer_model(training_features.shape[1])
    training_history = train_model(colette_transformer_model, training_features, training_labels)

    # Evaluate model
    evaluate_model(colette_transformer_model, test_features, test_labels)

    # Plot training history
    plot_training_history(training_history)

    # Real-time monitoring simulation (with random new data)
    incoming_data_stream = np.random.randn(5, training_features.shape[1])  # Simulating 5 new data points
    real_time_monitoring(colette_transformer_model, incoming_data_stream)

    # Explain model predictions using SHAP
    explain_model_shap(colette_transformer_model, training_features, test_features)
