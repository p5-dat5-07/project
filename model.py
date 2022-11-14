from json import load
import pretty_midi
import tensorflow as tf
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd

from load_data import load_data

NOTES_PER_DATA_ENTRY = 3
TRAINING_DATA_AMOUNT = 10

def main():
    data = load_data(TRAINING_DATA_AMOUNT, NOTES_PER_DATA_ENTRY)
    print(data)


def create_generator():
  input_shape = (seq_length, 3)

  inputs = tf.keras.Input(input_shape)
  x = tf.keras.layers.LSTM(128)(inputs)

  outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
  }

  model = tf.keras.Model(inputs, outputs)

  loss = {
    'pitch' : tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'step': mse_with_positive_pressure,
    'duration': mse_with_positive_pressure,
  }

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss=loss, optimizer=optimizer)

  return model

main()