# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:34:16 2024

@author: nigel
"""
from custom_transformer import TransformerBlock
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a synthetic dataset for binary classification (e.g., threat/no-threat)
def create_dataset(n_samples=1000, n_features=20, random_state=42):
    features, labels = make_classification(n_samples=n_samples, n_features=n_features, 
                                           n_informative=15, n_classes=2, 
                                           random_state=random_state)
    return features, labels

# Step 2: Preprocess the dataset
def preprocess_data(features, labels, test_size=0.3, random_state=42):
    """Preprocess the dataset by scaling and splitting."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

# Step 3: Build the Transformer-based model
def build_transformer_model(input_shape, num_heads=2, key_dim=64, ff_dim=128, learning_rate=1e-4):
    """Build and compile a transformer-based neural network model."""
    
    inputs = layers.Input(shape=(input_shape,))
    
    # Add a dense layer to project input to transformer-compatible shape
    x = layers.Dense(key_dim)(inputs)
    x = layers.Reshape((1, key_dim))(x)  # Reshape input to (batch_size, 1, key_dim) as input to Transformer

    # Add the Transformer block
    transformer_block = TransformerBlock(embed_dim=key_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block(x)
    
    # Flatten and add final dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification output
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Step 4: Train the model with early stopping
def train_model(model, X_train, y_train, epochs=30, batch_size=32, validation_split=0.2):
    """Train the model with a learning rate scheduler and early stopping."""
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=validation_split, 
                        callbacks=[lr_scheduler, early_stopping])
    return history

# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    return accuracy

# Step 6: Predict threats using the trained model
def predict_threats(model, new_data):
    """Make predictions on new, unseen data."""
    predictions = model.predict(new_data)
    return predictions

# Step 7: Plot training history
def plot_training_history(history):
    """Plot training & validation accuracy and loss."""
    plt.figure(figsize=(12, 6))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Step 8: Add Evaluation Metrics (NEW)
def evaluate_model_metrics(y_test, predictions, threshold=0.5):
    # Apply threshold to convert probabilities to binary predictions
    y_pred = (np.array(predictions) >= threshold).astype(int)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Classification Report (Precision, Recall, F1-Score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Threat", "Threat"]))

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # Step 1: Create dataset
    features, labels = create_dataset()

    # Step 2: Preprocess dataset
    X_train, X_test, y_train, y_test = preprocess_data(features, labels)

    # Step 3: Build the transformer-based model
    colette_transformer_model = build_transformer_model(X_train.shape[1])

    # Step 4: Train the model with early stopping
    history = train_model(colette_transformer_model, X_train, y_train)

    # Step 5: Evaluate the model
    evaluate_model(colette_transformer_model, X_test, y_test)

    # Step 6: Make predictions on new unseen data
    new_data = np.random.randn(5, 20)  # Random new data for testing
    threat_predictions = predict_threats(colette_transformer_model, new_data)
    print(f"Threat Predictions: {threat_predictions}")
    
    # Step 7: Plot the training history
    plot_training_history(history)
