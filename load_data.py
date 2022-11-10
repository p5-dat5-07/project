import pretty_midi
import tensorflow as tf
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd

step_in_sec = 16 * 4
NUM_FILES = 3
INPUT_NOTES = 8
OUTPUT_NOTES = 20
MAX_EPOCHS = 3

def compile_and_fit(model, data):
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    history = model.fit(data, epochs=MAX_EPOCHS)

    return history

class WindowGenerator():
    def __init__(self, input_width, output_width, data):
        self.data = data
        self.input_width = input_width
        self.output_width = output_width

        self.size = input_width + output_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.size)[self.input_slice]
        self.output_slice = slice(input_width, self.size)
        self.output_indices = np.arange(self.size)[self.output_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.size}',
            f'Input indices    : {self.input_indices}',
            f'Output indices   : {self.output_indices}'])

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_notes):
        super().__init__()
        self.out_notes = out_notes
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(3)
    
    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state
    
    def call(self, inputs):
        predictions = []
        pred, state = self.warmup(inputs)
        predictions.append(pred)

        for n in range(1, self.out_notes):
            x = prediction
            x = self.lstm_cell(x)
            prediction = self.dense(x)
            predictions.append(prediction)
        
        predictions = tf.stack(predictions)
        return predictions

class FeedbackGenerator(tf.keras.Model):
    def __init__(self, units,):
        j

def main():
    data = load_data(NUM_FILES, INPUT_NOTES)


def load_data(num_files: int, num_notes: int):
    data_dir = pathlib.Path('data/maestro-v2.0.0')
    if not data_dir.exists():
        tf.keras.utils.get_file(
        'maestro-v2.0.0-midi.zip',
        origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
        extract=True,
        cache_dir='.', cache_subdir='data',
    )
    files = glob.glob(str(data_dir/'**/*.mid*'))
    print('Number of files: ', len(files))

    data = []
    
    for f in files[:num_files]:
        notes = midi_to_notes(f, num_notes)
        print(notes.shape)
        data.append(notes)
    

    key_order = ['pitch', 'start', 'end', 'step', 'duration']
    data = pd.concat(data)

    train_notes = np.stack([data[key] for key in key_order], axis=1)

    dataset = tf.data.Dataset.from_tensor_slices(train_notes)

    return dataset

def sequences(dataset: tf.data.Dataset, num_input: int, num_output: int, vocab_size = 128) -> tf.data.Dataset:
    total_size = num_input + num_output
    

def midi_to_notes(file: str, num_notes: int) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(file)
    instrument = pm.instruments[0]
    instrument.notes.sort(key=lambda note: note.start)
    prev_start = instrument.notes[0].start

    notes = collections.defaultdict(list)
    for n in instrument.notes[0:num_notes]:
        start = n.start
        end = n.end
        notes['pitch'].append(n.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append((start - prev_start) * step_in_sec)
        notes['duration'].append((end - start) * step_in_sec)
        prev_start = start
    
    return pd.DataFrame.from_dict(notes)

def plot_piano_roll(notes: pd.DataFrame):
    title = f'Whole track'
    count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)
    plt.show()    


main()