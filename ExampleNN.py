import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

# Collects datasets from api
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'maestro-v2.0.0-midi.zip',
      origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
      extract=True,
      cache_dir='.', cache_subdir='data',
  )

filenames = glob.glob(str(data_dir/'**/*.mid*'))
print('Number of files:', len(filenames))

learning_rate = 0.05 # Learningrate
seq_length = 25 # Lenght of every sequence
batch_size = 64 # Batchsize
epochs = 1 # Epochs
vocab_size = 128 # Amount of possible pitches
num_files = 5 # Number og files for traning

temperature = 3.0
num_predictions = 75

key_order = ['pitch', 'step', 'duration'] #The order of the inputs in the input

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def main():
  LSTM_model = create_model()

  LSTM_model.summary()

  dataset, len_notes = load_data(num_files=num_files)

  seq_dataset = create_sequences(dataset=dataset, seq_length=seq_length, vocab_size=vocab_size)

  buffer_size = len_notes - seq_length  # the number of items left in the dataset (65)
  train_ds = (seq_dataset
              .shuffle(buffer_size)
              .batch(batch_size, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))

  disc_model = create_discriminator()

  disc_model.summary()

  train(train_ds, 'twinkle-twinkle-little-star.mid', LSTM_model=LSTM_model, disc_model=disc_model)

  generate_notes(LSTM_model, 'twinkle-twinkle-little-star.mid')

  '''
  train_model(LSTM_model, seq_dataset, len_notes)

  generate_notes(LSTM_model, 'twinkle-twinkle-little-star.mid')
  '''



# ------------ Functions ------------
#@tf.function
def train_step(midi_string, LSTM_model, disc_model):
  LSTM_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  disc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  raw_notes = midi_to_notes(midi_string)
  sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

  # The initial sequence of notes; pitch is normalized similar to training
  # sequences
  input_notes = (
    sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

  non_generated_train_notes = tf.expand_dims(input_notes, 0)  

  
  with tf.GradientTape() as LSTM_tape, tf.GradientTape() as disc_tape:
    # Generate and reformat notes
    generated_notes = LSTM_model(non_generated_train_notes, training=True)

    # Pitch divide vecab_size, since thats the format of the input
    generated_notes = tf.concat([tf.divide(generated_notes['pitch'], [vocab_size]), generated_notes['step'], generated_notes['duration']], -1)
    generated_notes = tf.slice(generated_notes, [0, 0], [1,75])
    generated_notes = tf.reshape(generated_notes, (seq_length, 3))

    #LSTM_loss = mse_with_positive_pressure(generated_notes['pitch'], tf.ones_like(generated_notes['pitch']))
    
    #generated_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    
    #generated_notes = (
    #  generated_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_train_notes = tf.expand_dims(generated_notes, 0)

    real_output = disc_model(non_generated_train_notes, training=True)
    fake_output = disc_model(generated_train_notes, training=True)

    LSTM_loss = generator_loss(fake_output=fake_output)
    disc_loss = discriminator_loss(real_output=real_output, fake_output=fake_output)

    print("LSTM LOSS: ", LSTM_loss)
    print("DISC LOSS: ", disc_loss)
    
  
    gradient_LSTM = LSTM_tape.gradient(LSTM_loss, LSTM_model.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)

    #print("GRADIENT LSTM: ", gradient_LSTM)
    #print("GRADIENT DISC: ", gradient_disc)

  LSTM_optimizer.apply_gradients(zip(gradient_LSTM, LSTM_model.trainable_variables))
  disc_optimizer.apply_gradients(zip(gradient_disc, disc_model.trainable_variables))

  #print("opt LSTM: ", LSTM_optimizer)
  #print("opt DISC: ", disc_optimizer)
  
  


def train(train_dataset, midi_string, LSTM_model, disc_model):
  for epoch in range(epochs):
    for i, batch in enumerate(train_dataset):
      print("Itteration: ", i)
      train_step(midi_string=midi_string, LSTM_model=LSTM_model, disc_model=disc_model)

    

def create_discriminator():
  input_shape = (seq_length, 3) 

  inputs = tf.keras.Input(input_shape) 
  x = tf.keras.layers.LSTM(128)(inputs)

  outputs = tf.keras.layers.Dense(1)(x)

  model = tf.keras.Model(inputs, outputs)

  return model

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  #print(total_loss)
  return total_loss

def generate_notes(model, midi_string):
  instrument = pretty_midi.PrettyMIDI(midi_string).instruments[0]
  instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

  raw_notes = midi_to_notes(midi_string)
  generated_notes = []

  sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

  # The initial sequence of notes; pitch is normalized similar to training
  # sequences
  input_notes = (
    sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))
  
  prev_start = 0
  # Generates
  for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

  generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end'))

  print(generated_notes.head(10))

  
  out_file = 'output.mid'
  out_pm = notes_to_midi(
    generated_notes, out_file=out_file, instrument_name=instrument_name)
  
  #return np.stack([generated_notes[key] for key in key_order], axis=1)
  
def predict_next_note(
    notes: np.ndarray, 
    model: tf.keras.Model, 
    temperature: float = 1.0) -> int:
  """Generates a note IDs using a trained sequence model."""

  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)

def train_model(model, seq_dataset, len_notes):
  buffer_size = len_notes - seq_length  # the number of items left in the dataset (65)
  train_ds = (seq_dataset
              .shuffle(buffer_size)
              .batch(batch_size, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))

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

  history = model.fit(
      train_ds,
      epochs=epochs,
      callbacks=callbacks,
  ) 
  model.save('homeMade.h5')

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
  
def create_model():
  input_shape = (seq_length, 3)  

  inputs = tf.keras.Input(input_shape)
  x = tf.keras.layers.LSTM(128)(inputs)
  print(x)

  outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
  }

  model = tf.keras.Model(inputs, outputs)
  '''
  loss = {
    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    'step': mse_with_positive_pressure,
    'duration': mse_with_positive_pressure,
  }

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss=loss, optimizer=optimizer)
  '''
  return model

def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size = 128) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)
  print(windows)
  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)
  print(sequences.element_spec)

  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return inputs, labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

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
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

main()