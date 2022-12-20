import os
import collections
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple
def main():
  #filenames = glob.glob(str('C:/Users/signe/OneDrive/Skrivebord/P5/project/final_models2/large-advanced/epoch20/*.mid*'))
  #print('Number of files:', len(filenames))
  max = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
  for dirname, dirs, files in os.walk(os.getcwd() + '/final_models4'):
    for i, filename in enumerate(files):
      filename_without_extension, extension = os.path.splitext(filename)
      if extension == '.mid':
        pm = pretty_midi.PrettyMIDI(os.path.join(dirname, filename))
        raw_notes = midi_to_notes(os.path.join(dirname, filename))
        if raw_notes.iloc[-1].end > max[filename_without_extension[-1]]:
          max[filename_without_extension[-1]] = (raw_notes.iloc[-1].end)
  print(max)
  
  for dirname, dirs, files in os.walk(os.getcwd() + '/final_models4'):
    for i, filename in enumerate(files):
      filename_without_extension, extension = os.path.splitext(filename)
      if extension == '.mid':
        pm = pretty_midi.PrettyMIDI(os.path.join(dirname, filename))
        
        instrument = pm.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

        raw_notes = midi_to_notes(os.path.join(dirname, filename))

        fig = plot_piano_roll(raw_notes, max=max[filename_without_extension[-1]])
        #plt.show()
        fig.savefig(f"{os.path.join(dirname, filename_without_extension)}.svg", format='svg')
  

def plot_piano_roll(notes: pd.DataFrame, max: int, count: Optional[int] = None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  figure = plt.figure(figsize=(20, 4))
  plt.xlim([0, max])
  plt.ylim([0, 120])
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)
  return figure

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