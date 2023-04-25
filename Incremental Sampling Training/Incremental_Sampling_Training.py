import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load and preprocess data (X_train, y_train)
# ...

# Shuffle dataset
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

# Model definition
model = Sequential([
    Dense(64, activation='relu', input_shape=X_train.shape[1:]),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# Model training parameters
initial_lr = 0.001
min_lr = 0.00001
lr_decay = 0.9
batch_size = 32
initial_samples = 100
increment = 50

# Incremental Sampling and Training
n_samples = len(X_train)
samples_used = initial_samples
optimizer = Adam(lr=initial_lr)

while samples_used <= n_samples:
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train[:samples_used], y_train[:samples_used], batch_size=batch_size, epochs=1, verbose=0)

    # Update learning rate and increment samples
    optimizer.lr = max(min_lr, optimizer.lr * lr_decay)
    samples_used += increment

# Evaluate model
# ...
