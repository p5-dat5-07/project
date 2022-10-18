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

class Categorical(tf.keras.layers.Layer):
  def __init__(self, input_dim):
    super(Categorical, self).__init__()
    # Create a non-trainable weight.
    self.total = tf.Variable([[0]], shape=[None,1], dtype=tf.float32,
                               trainable=False)

  def call(self, inputs):
    self.total.assign(tf.cast(tf.random.categorical(inputs, num_samples=1), dtype=tf.float32))
    return self.total


# Collects datasets from api
data_dir = pathlib.Path('data/q-maestro-v2.0.0')

filenames = glob.glob(str(data_dir/'**/*.mid*'))
print('Number of files:', len(filenames))

learning_rate = 0.01 # Learningrate
seq_length = 75 # Lenght of every sequence
batch_size = 32 # Batchsize
epochs = 1 # Epochs
vocab_size = 128 # Amount of possible pitches
num_files = 5 # Number og files for traning

temperature = 3.0
num_predictions = 75
step_in_sec = 16 * 4

key_order = ['pitch', 'step', 'duration'] #The order of the inputs in the input

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#print(tf.constant([1., 2., 3., 4.]))
#print(Categorical()(tf.constant([1., 2., 3., 4.])))

def main():
  LSTM_model = create_LSTM_model()
  LSTM_model.summary()

  disc_model = create_discriminator()
  disc_model.summary()

  dataset, len_notes = load_data(num_files=num_files)

  seq_dataset = create_sequences(dataset=dataset, seq_length=seq_length, vocab_size=vocab_size)

  buffer_size = len_notes - seq_length  # the number of items left in the dataset 
  train_ds = (seq_dataset
              .shuffle(buffer_size)
              .batch(batch_size, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))
  
  train(train_dataset=train_ds, LSTM_model=LSTM_model, disc_model=disc_model)


#@tf.function
def train_step(batch, label, LSTM_model, disc_model):
  LSTM_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  disc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  with tf.GradientTape() as LSTM_tape, tf.GradientTape() as disc_tape:
    LSTM_tape.watch(batch)
    input_notes = batch = tf.cast(batch, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)

    predictions = LSTM_model(input_notes, training=True)
    
    batch = tf.slice(batch, [0, 1, 0], [batch_size, seq_length - 1, 3])
    generated_notes = tf.concat([batch, tf.expand_dims(predictions, 1)], axis=1)
    batch = tf.concat([batch, tf.expand_dims(label, 1)], axis=1)

    real_output = disc_model(batch, training=True)
    fake_output = disc_model(generated_notes, training=True)

    print("BATCH: ", batch[0, seq_length-1])
    print("NODE: ", generated_notes[0, seq_length-1])

    #print("REAL OUT: ", real_output[0])
    #print("FAKE OUT: ", fake_output[0])

    LSTM_loss = generator_loss(fake_output=fake_output)
    disc_loss = discriminator_loss(real_output=real_output, fake_output=fake_output)

    print("LSTM LOSS: ", LSTM_loss)
    print("DISC LOSS: ", disc_loss)
    
  
  gradient_LSTM = LSTM_tape.gradient(LSTM_loss, LSTM_model.trainable_variables)
  gradient_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)

  LSTM_optimizer.apply_gradients(zip(gradient_LSTM, LSTM_model.trainable_variables))
  disc_optimizer.apply_gradients(zip(gradient_disc, disc_model.trainable_variables))
  

def train(train_dataset, LSTM_model, disc_model):
  i = 0
  for epoch in range(epochs):
    for batch, target in train_dataset:
      print("Itteration: ", i)
      i += 1
      train_step(batch=batch, label=target, LSTM_model=LSTM_model, disc_model=disc_model)

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size = 128) -> tf.data.Dataset:
  # Take 1 extra for the labels
  windows = dataset.window(seq_length + 1, shift=1, stride=1,
                              drop_remainder=True)
                       
  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length + 1, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels = sequences[-1]

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def create_discriminator():
  input_shape = (seq_length, 3) 

  inputs = tf.keras.Input(input_shape) 
  x = tf.keras.layers.LSTM(128)(inputs)

  outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

  model = tf.keras.Model(inputs, outputs)

  return model

def create_LSTM_model():
  # seq_length is number of timesteps (how long is the sequence)
  # 3 is the number of fetures in every timestep (3 for "pitch", "step" and "duration")
  input_shape = (seq_length, 3)  

  inputs = tf.keras.Input(input_shape)
  x = tf.keras.layers.LSTM(128)(inputs)

  dense_layers = {
    'pitch': tf.keras.layers.Dense(128, name='pitch', activation="ReLU")(x),
    'step': tf.keras.layers.Dense(1, name='step', activation="LeakyReLU")(x),
    'duration': tf.keras.layers.Dense(1, name='duration', activation="LeakyReLU")(x),
  }

  pitchLayer = Categorical(128)(dense_layers['pitch'])
  output = tf.keras.layers.Concatenate(axis=1)([pitchLayer, dense_layers['step'], dense_layers['duration']])

  model = tf.keras.Model(inputs, output)

  return model

def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)
  print(notes)
  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append((start - prev_start) * step_in_sec)
    notes['duration'].append((end - start) * step_in_sec)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def load_data(num_files: int):
  all_notes = []
  for f in filenames[:num_files]:
    notes = midi_to_notes(f)
    all_notes.append(notes)

  all_notes = pd.concat(all_notes)
  train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

  notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
  notes_ds.element_spec 

  return notes_ds, len(all_notes)

main()
