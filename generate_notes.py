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

learning_rate = 0.001 # Learningrate
seq_length = 50 # Lenght of every sequence
batch_size = 32 # Batchsize
epochs = 50 # Epochs
vocab_size = 128 # Amount of possible pitches
num_files = 7 # Number og files for traning
off_set = 400 # Where to start with the data

temperature = 2.0
num_predictions = 200
step_in_sec = 16 * 4

key_order = ['pitch', 'step', 'duration'] #The order of the inputs in the input


data_dir = pathlib.Path('data/q-maestro-v2.0.0')

filenames = glob.glob(str(data_dir/'**/*.mid*'))
print('Number of files:', len(filenames))

def main():
  LSTM_model = tf.keras.models.load_model('MT_s50_b64_e50_f100_o100.h5', compile=False)
  generate_notes(LSTM_model, 'twinkle-twinkle-little-star.mid')


def generate_notes(model, midi_string):
  instrument = pretty_midi.PrettyMIDI(midi_string).instruments[0]
  instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

  raw_notes = midi_to_notes(midi_string)
  

  sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

  # The initial sequence of notes; pitch is normalized similar to training
  # sequences
  input_notes = (
    sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

  prev_start = 0
  generated_notes = []
  for x in input_notes:
    pitch, step, duration = x[0] * vocab_size, x[1], x[2]
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    prev_start = start
  
  # Generates
  for i in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    generated_notes.append((pitch, step, duration, start, end))
    input_note = (pitch, step - 0.5, duration - 0.5)
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(tf.divide(input_note, [128, 1, 1]), 0), axis=0)

    prev_start = start

  generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end'))

  print(generated_notes.to_string())

  
  out_file = '100ftw.mid'
  out_pm = notes_to_midi(
    generated_notes, out_file=out_file, instrument_name=instrument_name)

def predict_next_note(
    notes: np.ndarray, 
    keras_model: tf.keras.Model, 
    temperature: float = 1.0) -> int:
  """Generates a note IDs using a trained sequence model."""

  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = keras_model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step'] + 0.5
  duration = predictions['duration'] + 0.5

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)
  

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

main()