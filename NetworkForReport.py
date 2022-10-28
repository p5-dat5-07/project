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
batch_size = 32 # Batchsize
epochs = 20 # Epochs
vocab_size = 128 # Amount of possible pitches
num_files = 50 # Number og files for traning

input_shape = vocab_size
input = tf.keras.layers.Input(input_shape)
h1 = tf.keras.layers.Dense(30)(input)
h2 = tf.keras.layers.Dense(30)(h1)
output = tf.keras.layers.Dense(128)(h2)

model = tf.keras.Model(input, output)
model.summary()