import pandas as pd
import numpy as np
import pretty_midi
import glob
import pathlib
import collections
import json
import os
from consts import *
from args import Data, Params

class DataManager:
    file_names:     [str]
    key_order:      [str]
    settings:       Data
    params:         Params

    def __init__(self, params, settings):
        self.params         = params
        self.settings       = settings
        self.files          = glob.glob(str(pathlib.Path(self.settings.input)/"**/*.mid*"))
        self.file_names     = self.files[self.settings.offset:self.settings.offset+self.settings.amount]
        self.key_order      = ["pitch", "step", "duration"]

    def generate_dataset(self):
        data = None
        data_length = 0
        # Get data
        if self.params.sample_location + self.params.samples_per_epoch > len(self.files):
            print(f"Sample location ({self.params.sample_location}) + samples per epoch ({self.params.samples_per_epoch}) has to be lower than the total amount of files ({len(self.files)})!")
            exit()
        offset = self.settings.offset
        file_count = self.settings.amount
        for file in self.file_names:
            notes = self.midi_to_notes(file)
            train_notes = np.stack([notes[key] for key in self.key_order], axis=1)
            data_length += len(train_notes)
            notes_dataset = tf.data.Dataset.from_tensor_slices(train_notes)
            sequences = self.create_sequences(notes_dataset, file)
            if data == None:
                data = sequences
            else: 
                data = data.concatenate(sequences)
        # Shuffle data and batch data
        buffer_size = data_length - self.params.sequence_length  # the number of items left in the dataset 
        training_data = (data.shuffle(buffer_size)
                            .batch(self.params.batch_size, drop_remainder=True)
                            .cache()
                            .prefetch(tf.data.experimental.AUTOTUNE))
        
        training_data.save(self.settings.output)

    def create_sequences(self, dataset: tf.data.Dataset, file_path: str) -> [tf.data.Dataset]:
        """Returns TF Dataset of sequence and label examples."""
        sequence_length = self.params.sequence_length+1

        # Take 1 extra for the labels
        windows = dataset.window(sequence_length, shift=1, stride=1,
                                    drop_remainder=True)

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(sequence_length, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        # Normalize note pitch
        def scale_pitch(x):
            x = x/[self.params.vocab_size,1.0,1.0]
            return x

        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            labels_dense = sequences[-1]
            labels = {key:labels_dense[i] for i,key in enumerate(self.key_order)}
            labels['step'] += [0.001]
            labels['duration'] += [0.001]
            return scale_pitch(inputs), labels, self.get_key_in_filename(file_path)
        
        return sequences.map(split_labels)


    def midi_to_notes(self, file: str) -> pd.DataFrame:
        pm = pretty_midi.PrettyMIDI(file)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)
        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes["pitch"].append(note.pitch)
            notes["step"].append((start - prev_start) * self.params.steps_per_seconds)
            notes["duration"].append((end - start) * self.params.steps_per_seconds)
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


    def get_key_in_filename(self, file_path: str):
        with open("./keys.json") as file:
            jsonObject = json.load(file)
            file.close()

        for x in jsonObject:
            if (x["file_path"][5:-1] == file_path[8:-1]):
                return x["key"]


