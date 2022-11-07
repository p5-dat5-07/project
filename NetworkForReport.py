import tensorflow as tf
import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

learning_rate = 0.001 # Learningrate
seq_length = 50 # Lenght of every sequence
batch_size = 64 # Batchsize
epochs = 50 # Epochs
vocab_size = 128 # Amount of possible pitches
num_files = 100 # Number og files for traning
off_set = 100 # Where to start with the data



def create_LSTM_model():
  # seq_length is number of timesteps (how long is the sequence)
  # 3 is the number of fetures in every timestep (3 for "pitch", "step" and "duration")
  input_shape = (seq_length, 3)  

  inputs = tf.keras.Input(input_shape)
  x = tf.keras.layers.LSTM(128)(inputs)

  ph1 = tf.keras.layers.Dense(30, name='pitch_hidden1')(x)
  sh1 = tf.keras.layers.Dense(30, name='step_hidden1')(x)
  dh1 = tf.keras.layers.Dense(30, name='duration_hidden1')(x)

  ph2 = tf.keras.layers.Dense(30, name='pitch_hidden2')(ph1)
  sh2 = tf.keras.layers.Dense(30, name='step_hidden2')(sh1)
  dh2 = tf.keras.layers.Dense(30, name='duration_hidden2')(dh1)

  outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(ph2),
    'step': tf.keras.layers.Dense(1, name='step')(sh2),
    'duration': tf.keras.layers.Dense(1, name='duration')(dh2),
  }

  model = tf.keras.Model(inputs, outputs)

  return model

model = create_LSTM_model()
model.summary()