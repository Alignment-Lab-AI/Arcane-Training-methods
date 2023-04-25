import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam

# Load and preprocess data (X, y)
# X shape: (n_samples, seq_length, n_features)
# y shape: (n_samples, seq_length, n_outputs)
# ...

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Temporal aggregation function
def aggregate_temporal_data(data, aggregation_interval, aggregation_function=np.mean):
    return aggregation_function(data.reshape(-1, data.shape[1] // aggregation_interval, aggregation_interval, data.shape[2]), axis=2)

# Temporal aggregation
aggregation_interval = 4
X_train_agg = aggregate_temporal_data(X_train, aggregation_interval)
X_test_agg = aggregate_temporal_data(X_test, aggregation_interval)

# Model definition
def create_model(input_shape, output_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, activation='relu', return_sequences=True)),
        Bidirectional(LSTM(32, activation='relu', return_sequences=True)),
        TimeDistributed(Dense(output_shape, activation='softmax'))
    ])
    return model

# Model training parameters
lr = 0.001
batch_size = 32
epochs = 20

# Model training
model = create_model(X_train_agg.shape[1:], y_train.shape[2])
optimizer = Adam(lr=lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_agg, y_train[:, ::aggregation_interval, :], batch_size=batch_size, epochs=epochs, verbose=1)

# Inference and reverse mapping
def infer_and_reverse_map(model, X, aggregation_interval):
    X_agg = aggregate_temporal_data(X, aggregation_interval)
    y_pred_agg = model.predict(X_agg)

    # Reverse mapping
    y_pred = np.repeat(y_pred_agg, aggregation_interval, axis=1)
    return y_pred

# Model inference
y_pred = infer_and_reverse_map(model, X_test, aggregation_interval)

# Model evaluation
# Assuming y_test and y_pred are one-hot encoded
y_test_label = np.argmax(y_test, axis=2)
y_pred_label = np.argmax(y_pred, axis=2)

accuracy = accuracy_score(y_test_label.flatten(), y_pred_label.flatten())
print("Accuracy:", accuracy)
