import string
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
import 

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence

# Collects datasets from api
data_dir = pathlib.Path('data/q-maestro-v2.0.0')

filenames = glob.glob(str(data_dir/'**/*.mid*'))
print('Number of files:', len(filenames))

learning_rate = 0.001 # Learningrate
seq_length = 50 # Lenght of every sequence
batch_size = 64 # Batchsize
epochs = 50 # Epochs
vocab_size = 128 # Amount of possible pitches
num_files = 5 # Number og files for traning
num_eval_files = int(num_files / 5)
off_set = 50 # Where to start with the data
off_set_eval = 25

temperature = 3.0
num_predictions = 10
step_in_sec = 16 * 4

norm = 0.25

key_order = ['pitch', 'step', 'duration'] #The order of the inputs in the input

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
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
  0: tf.constant([0, 2, 4, 5, 7, 9, 11], dtype=tf.int64), 
  1: tf.constant([1, 3, 5, 6, 8, 10, 0], dtype=tf.int64),
  2: tf.constant([2, 4, 6, 7, 9, 11, 1], dtype=tf.int64),
  3: tf.constant([3, 5, 7, 8, 10, 0, 2], dtype=tf.int64),
  4: tf.constant([4, 6, 8, 9, 11, 1, 3], dtype=tf.int64),
  5: tf.constant([5, 7, 9, 10, 0, 2, 4], dtype=tf.int64),
  6: tf.constant([6, 8, 10, 11, 1, 3, 5], dtype=tf.int64),
  7: tf.constant([7, 9, 11, 0, 2, 4, 6], dtype=tf.int64), 
  8: tf.constant([8, 10, 0, 1, 3, 5, 7], dtype=tf.int64),
  9: tf.constant([9, 11, 1, 2, 4, 6, 8], dtype=tf.int64),
  10: tf.constant([10, 0, 2, 3, 5, 7, 9], dtype=tf.int64),
  11: tf.constant([11, 1, 3, 4, 6, 8, 10], dtype=tf.int64),
  12: tf.constant([0, 2, 3, 5, 7, 8, 10], dtype=tf.int64),
  13: tf.constant([1, 3, 4, 6, 8, 9, 11], dtype=tf.int64),
  14: tf.constant([2, 4, 5, 7, 9, 10, 0], dtype=tf.int64),
  15: tf.constant([3, 5, 6, 8, 10, 11, 1], dtype=tf.int64),
  16: tf.constant([4, 6, 7, 9, 11, 0, 2], dtype=tf.int64),
  17: tf.constant([5, 7, 8, 10, 0, 1, 3], dtype=tf.int64),
  18: tf.constant([6, 8, 9, 11, 1, 2, 4], dtype=tf.int64),
  19: tf.constant([7, 9, 10, 0, 2, 3, 5], dtype=tf.int64),
  20: tf.constant([8, 10, 11, 1, 3, 4, 6], dtype=tf.int64),
  21: tf.constant([9, 11, 0, 2, 4, 5, 7], dtype=tf.int64),
  22: tf.constant([10, 0, 1, 3, 5, 6, 8], dtype=tf.int64),
  23: tf.constant([11, 1, 2, 4, 6, 7, 9], dtype=tf.int64),
}

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def get_key_in_filename(filename):
  with open("keys.json") as jf:
    jsonObject = json.load(jf)
    jf.close()

  for x in jsonObject:
    if (x['file_path'][5:-1] == filename[8:-1]):
      print("found")
      return key_dict[x['key']]

def main():
  LSTM_model = create_LSTM_model()
  LSTM_model.summary()

  # Loading all the data into datasets of sequences with 3 fetures[pitch, step, duration] shape(seq_length, 3)
  all_len = 0
  for i, f in enumerate(filenames[off_set:num_files+off_set]):
    dataset, len_notes = load_data(f)
    all_len = all_len + len_notes
    seq_dataset = create_sequences(dataset=dataset, seq_length=seq_length, vocab_size=vocab_size, filepath=f)

    if (i == 0):
      collected_ds = seq_dataset
    else:
      collected_ds = collected_ds.concatenate(seq_dataset)

  # Shuffle data and divide into batches
  buffer_size = all_len - seq_length  # the number of items left in the dataset 
  train_ds = (collected_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))
  
  # Loading all the data into datasets of sequences with 3 fetures[pitch, step, duration] shape(seq_length, 3)
  all_len_eval = 0
  for i, f in enumerate(filenames[off_set + num_files + off_set_eval:off_set + num_files + off_set_eval + num_eval_files]):
    dataset, len_notes = load_data(f)
    all_len_eval = all_len_eval + len_notes
    seq_dataset = create_sequences(dataset=dataset, seq_length=seq_length, vocab_size=vocab_size, filepath=f)

    if (i == 0):
      collected_ds = seq_dataset
    else:
      collected_ds = collected_ds.concatenate(seq_dataset)

  # Shuffle data and divide into batches
  buffer_size_eval = all_len - seq_length  # the number of items left in the dataset 
  eval_ds = (collected_ds
            .shuffle(buffer_size_eval)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))

  
  '''
  # Random testdata
  step = 500
  data_x = []
  data_y = []
  for i in range(step):
    data_x = (tf.random.uniform([batch_size,128], minval=0))
    rand_p = tf.random.uniform([batch_size], minval=0, maxval=128, dtype=tf.int32)
    rand_p = tf.cast(rand_p, dtype=tf.float32)
    data_y.append((data_x, {'pitch': rand_p, 'step': tf.random.uniform([batch_size], minval=0), 'duration': tf.random.uniform([batch_size], minval=0)}))
    #data_y.append((data_x, tf.random.uniform([batch_size, 128])))
  '''
  prev_eval_dict = {'pitch': 1000, 'step': 1000, 'duration': 1000}
  prev_train_dict = {'pitch': 1000, 'step': 1000, 'duration': 1000}
  # Train for epochs
  for epoch in range(epochs):
    print("epoch: ", epoch)
    train_dict = train(LSTM_model, train_ds)
    eval_dict = eval(LSTM_model, eval_ds)
    if (epoch > 5 and sum(prev_eval_dict.values()) < sum(eval_dict.values()) and sum(prev_eval_dict.values()) - sum(eval_dict.values()) < sum(prev_train_dict.values()) - sum(train_dict.values())):
      break
    prev_eval_dict = eval_dict
    prev_train_dict = train_dict

  LSTM_model.save("MT.h5")

def eval(model, eval_ds):
  all_pitch_loss = 0.0
  all_step_loss = 0.0
  all_duration_loss = 0.0
  step = 0

  for step, (x_batch_eval, y_batch_eval, keys) in enumerate(eval_ds):
    logits = model(x_batch_eval)

    pitch_loss = pitch_loss_mt(y_batch_eval['pitch'], logits['pitch'], keys) * 0.1
    step_loss = mse_with_positive_pressure(y_batch_eval['step'], logits['step'])
    duration_loss = mse_with_positive_pressure(y_batch_eval['duration'], logits['duration'])

    # Adding loss for print
    all_pitch_loss =  all_pitch_loss + pitch_loss 
    all_step_loss = all_step_loss + step_loss 
    all_duration_loss = all_duration_loss + duration_loss 

    step = step

  return {'pitch': all_pitch_loss / (step + 1), 'dration': all_duration_loss / (step + 1), 'step': all_step_loss / (step + 1)}


@tf.function
def train(model, train_ds):
  all_pitch_loss = 0.0
  all_step_loss = 0.0
  all_duration_loss = 0.0
  step = tf.cast(0, dtype=tf.int64)

  for step, (x_batch_train, y_batch_train, keys) in enumerate(train_ds):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.

    with tf.GradientTape() as tape:
      # Run the forward pass of the layer.
      # The operations that the layer applies
      # to its inputs are going to be recorded
      # on the GradientTape.
      logits = model(x_batch_train, training=True)  # Logits for this minibatch
 
      # Compute the loss value for this minibatch.
      pitch_loss = pitch_loss_mt(y_batch_train['pitch'], logits['pitch'], keys) * 0.1
      step_loss = mse_with_positive_pressure(y_batch_train['step'], logits['step'])
      duration_loss = mse_with_positive_pressure(y_batch_train['duration'], logits['duration'])

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.

    gradients = tape.gradient([pitch_loss, step_loss, duration_loss], model.trainable_variables)
    grads = [tf.clip_by_norm(g, norm)
             for g in gradients]

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Adding loss for print
    all_pitch_loss =  all_pitch_loss + pitch_loss 
    all_step_loss = all_step_loss + step_loss 
    all_duration_loss = all_duration_loss + duration_loss 

    step = step
    # Log every 200 batches.
    if step % 100 == 0:
      step = tf.cast(step, dtype=tf.float32)
      avg_pitch_loss = all_pitch_loss / (step + 1)
      avg_step_loss = all_step_loss / (step + 1)
      avg_duration_loss = all_duration_loss / (step + 1)
      tf.print(
        "Training loss (avg) at step ", step, " - Loss: ", avg_pitch_loss + avg_step_loss + avg_duration_loss, " - pitch: ", avg_pitch_loss, ", step: ", avg_step_loss, ", duration: ", avg_duration_loss)
  
  step = tf.cast(step, dtype=tf.float32)
  return {'pitch': all_pitch_loss / (step + 1), 'dration': all_duration_loss / (step + 1), 'step': all_step_loss / (step + 1)}
    #if step%100==0:
    #  print(loss.numpy())


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  y_true = tf.cast(y_true, dtype=tf.float32)
  mse = (y_true - y_pred) ** 2
  #positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse)

def pitch_loss_mt(y_true: tf.Tensor, y_pred: tf.Tensor, current_keys: tf.Tensor):
  succ = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True) 
  
  if current_keys == None:
    current_keys = tf.constant([0,1,2,3,4,5,6,7,8,9,10,11], dtype=tf.int64)

  index = tf.random.categorical(y_pred, num_samples=1)
  y = tf.math.floormod(index, tf.constant(12, dtype=tf.int64))
  equal = tf.math.equal(y, current_keys)

  f = tf.reduce_any(equal, 1)
  tr = tf.fill((batch_size, 1), 1.0)
  return succ(y_true, y_pred) + 0.1 * cross_entropy(f, tr)

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

def create_sequences(
    dataset: tf.data.Dataset, 
    seq_length: int,
    filepath: string,
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
    x = x/[vocab_size,1.0,1.0] - [0, 0.5, 0.5]
    return x

  def normalize(x):
    x['step'] = x['step'] - [0.5]
    x['duration'] = x['duration'] - [0.5]
    return {'pitch': x['pitch'], 'step': x['step'], 'duration': x['duration'],}

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), normalize(labels), get_key_in_filename(filepath)
  
  return sequences.map(split_labels)

def load_data(file: string):
  notes = midi_to_notes(file)

  #notes = pd.concat(notes)
  train_notes = np.stack([notes[key] for key in key_order], axis=1)

  notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
  notes_ds.element_spec 

  return notes_ds, len(notes)

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
    notes['step'].append((start - prev_start))
    notes['duration'].append((end - start))
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

main()