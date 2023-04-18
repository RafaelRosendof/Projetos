# -*- coding: utf-8 -*-
"""RedesLSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17lWqtgjwp8wWsbW87Dmeb15UoHfqAA_y
"""

#rede neural para a base IMDB

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers, optimizers, losses, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D

from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential

# constants and hyperparameters
MAX_WORD_INDEX = 10000

BATCH_SIZE = 128
NUM_EPOCHS = 32
LR = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1.0e-8
DECAY = 0.0
VAL_PERC = 0.9

EMBEDDING_DIM = 32
NUM_LSTM_UNITS = 32
DROPOUT_RATE = 0.5

# load database using Keras
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = MAX_WORD_INDEX)

#  print some information on the data
max_seq_len_train = max([len(sequence) for sequence in train_data])
max_seq_len_test = max([len(sequence) for sequence in test_data])
min_seq_len_train = min([len(sequence) for sequence in train_data])
min_seq_len_test = min([len(sequence) for sequence in test_data])
print(f'Maximum train sequence length: {max_seq_len_train}')
print(f'Maximum test sequence length: {max_seq_len_test}')
print(f'Minimum train sequence length: {min_seq_len_train}')
print(f'Minimum test sequence length: {min_seq_len_test}')

# randomly selects a sentence, look at the encoding and check its label
word_index = imdb.get_word_index()

ind = 33

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[ind]])

print(f'REVIEW:\n {decoded_review}\n')
print(f'Encoded sequence of words:\n {train_data[ind]}\n')
print(f'Label: {train_labels[ind]}\n')

# pad sequences
X_train = keras.preprocessing.sequence.pad_sequences(train_data)
X_test = keras.preprocessing.sequence.pad_sequences(test_data)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# transform labels into arrays
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# split training data into training and validation
nsamples = X_train.shape[0]
nval_samples = int(VAL_PERC * nsamples)
X_val = X_train[:nval_samples]
partial_X_train = X_train[nval_samples:]
y_val = y_train[:nval_samples]
partial_y_train = y_train[nval_samples:]

model = models.Sequential()
model.add(layers.Embedding(MAX_WORD_INDEX, EMBEDDING_DIM))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=1, activation='sigmoid'))

# set optimizer
opt = optimizers.Adam(learning_rate=LR,
                      beta_1=BETA1,
                      beta_2=BETA2,
                      epsilon=EPSILON,
)
# set loss and metrics
loss = losses.binary_crossentropy
met = [metrics.binary_accuracy]

# compile model: optimization method, training criterion and metrics
model.compile(
    optimizer=opt,
    loss=loss,
    metrics=met
)

#parada 
callbacks_list=[
    EarlyStopping(
        monitor ='binary_accuracy',
        patience=10
    ),
]

# train model
history = model.fit(partial_X_train,
                    partial_y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list,
                    verbose=1)

# learning curves
history_dict = history.history
history_dict.keys()

# losses
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

# accuracies
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

epochs = range(NUM_EPOCHS)

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))

ax1.plot(epochs, loss_values, 'bo', label="Training Loss")
ax1.plot(epochs, val_loss_values, 'b', label="Validation Loss")
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss Value')
ax1.legend()

ax2.plot(epochs, acc_values, 'ro', label="Training Accuracy")
ax2.plot(epochs, val_acc_values, 'r', label="Validation Accuracy")
ax2.set_title('Training and Validation Accuraccy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()

