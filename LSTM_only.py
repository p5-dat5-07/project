import tensorflow as tf
import json
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

learning_rate = 0.001 # Learningrate
seq_length = 75 # Lenght of every sequence
batch_size = 32 # Batchsize
epochs = 20 # Epochs
vocab_size = 128 # Amount of possible pitches
num_files = 1 # Number og files for traning

temperature = 3.0
num_predictions = 10
step_in_sec = 16 * 4

key_order = ['pitch', 'step', 'duration'] #The order of the inputs in the input
'''
key_dict = {
  0: C[0,2,4,5,7,9,11], 
  1: Db[1, 3, 5, 6, 8, 10, 0], 
  2: D[2, 4, 6, 7, 9,11,1], 
  3: Eb[3, 5, 7, 8, 10, 0, 2], 
  4: E[4, 6, 8, 9, 11, 1, 3], 
  5: F[5, 7, 9, 10, 0, 2, 4, 4], 
  6: Fsharp[6, 8, 10, 11, 1, 3, 5, 6], 
  7: G[7, 9, 11, 0, 2, 4, 6, 7], 
  8: Ab[8, 10, 0, 1, 3, 5, 7], 
  9: A[9, 11, 1, 2, 4, 6, 8, 9], 
  10: Bb[10, 0, 2, 3, 5, 7, 9, 10], 
  11: B[11, 1, 3, 4, 6, 8, 10, 11], 
  12: c[0, 2, 3, 5, 7, 8, 10], 
  13: csharp[1, 3, 4, 6, 8, 9, 11], 
  14: d[2, 4, 5, 7, 9, 10, 0], 
  15: eb[3, 5, 6, 8, 10, 11, 1], 
  16: e[4, 6, 7, 9, 11, 0, 2], 
  17: f, 
  18: fsharp, 
  19: g, 
  20: ab, 
  21: a, 
  22: bb, 
  23: b
}
'''

key_dict = {
  0: [0,2,4,5,7,9,11], 
  1: [1, 3, 5, 6, 8, 10, 0], 
  2: [2, 4, 6, 7, 9,11,1], 
  3: [3, 5, 7, 8, 10, 0, 2], 
  4: [4, 6, 8, 9, 11, 1, 3], 
  5: [5, 7, 9, 10, 0, 2, 4], 
  6: [6, 8, 10, 11, 1, 3, 5], 
  7: [7, 9, 11, 0, 2, 4, 6], 
  8: [8, 10, 0, 1, 3, 5, 7], 
  9: [9, 11, 1, 2, 4, 6, 8, 9], 
  10: [10, 0, 2, 3, 5, 7, 9, 10], 
  11: [11, 1, 3, 4, 6, 8, 10, 11], 
  12: [0, 2, 3, 5, 7, 8, 10], 
  13: [1, 3, 4, 6, 8, 9, 11], 
  14: [2, 4, 5, 7, 9, 10, 0], 
  15: [3, 5, 6, 8, 10, 11, 1], 
  16: [4, 6, 7, 9, 11, 0, 2], 
}

def get_key_in_filename(filename):
  with open("keys.json") as jf:
    jsonObject = json.load(jf)
    jf.close()

  for x in jsonObject:
    if (x['file_path'][2:-1] == filename[5:-1]):
      return key_dict[x['key']]

def main():
  index = tf.random.categorical(tf.constant([[0.9,0.1,0.0, 0.0, 0.0], [0.3,0.6,0.1,0.0,0.0]]), num_samples=1)

  y = tf.math.floormod(index, tf.constant(12, dtype=tf.int64))

  
  print(y)

  current_keys = get_key_in_filename(filenames[0])
  print(current_keys)
  
  LSTM_model = create_LSTM_model()
  LSTM_model.summary()

  dataset, len_notes = load_data(num_files=num_files)

  seq_dataset = create_sequences(dataset=dataset, seq_length=seq_length, vocab_size=vocab_size)

  buffer_size = len_notes - seq_length  # the number of items left in the dataset 
  train_ds = (seq_dataset
              .shuffle(buffer_size)
              .batch(batch_size, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))

  loss = {
    'pitch': pitch_loss_mt,
    'step': mse_with_positive_pressure,
    'duration': mse_with_positive_pressure,
  }

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  LSTM_model.compile(
    loss=loss, 
    loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration':1.0,}, 
    optimizer=optimizer)

  LSTM_model.evaluate(train_ds, return_dict=True)
  
  callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
  ]

  history = LSTM_model.fit(
      train_ds,
      epochs=epochs,
      callbacks=callbacks,
  ) 

  LSTM_model.save('LSTM.h5')

  LSTM_model.evaluate(train_ds, return_dict=True)
  


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * step_in_sec * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def pitch_loss_mt(y_true: tf.Tensor, y_pred: tf.Tensor):
  succ = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

  index = tf.random.categorical(y_pred, num_samples=1)

  y = tf.math.floormod(index, tf.constant(12, dtype=tf.int64))
  
  return succ(y_true, y_pred) 

def create_LSTM_model():
  # seq_length is number of timesteps (how long is the sequence)
  # 3 is the number of fetures in every timestep (3 for "pitch", "step" and "duration")
  input_shape = (seq_length, 3)  

  inputs = tf.keras.Input(input_shape)
  x = tf.keras.layers.LSTM(128)(inputs)

  dense_layers = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

  #pitch_logic = Categorical(128)(dense_layers['pitch'])
  output = tf.keras.layers.Concatenate(axis=1)([dense_layers['pitch'], dense_layers['step'], dense_layers['duration']])

  model = tf.keras.Model(inputs, dense_layers)

  return model

def create_sequences(
    dataset: tf.data.Dataset, 
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels
  
  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)



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

main()