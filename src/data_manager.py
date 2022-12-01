import numpy as np
import glob
import pathlib
import collections
import json
import os
from consts import *
from args import Data, Params
from util import midi_to_notes
loss_const = 0.2
chords = [[[0, 4, 7, 11], [2, 5, 9, 0], [4, 7, 11, 2], [5, 9, 0, 4], [7, 11, 2, 5], [9, 0, 4, 7], [11, 2, 5, 9]],
[[1, 5, 8, 0], [3, 6, 10, 1], [5, 8, 0, 3], [6, 10, 1, 5], [8, 0, 3, 6], [10, 1, 5, 8], [0, 3, 6, 10]],
[[2, 6, 9, 1], [4, 7, 11, 2], [6, 9, 1, 4], [7, 11, 2, 6], [9, 1, 4, 7], [11, 2, 6, 9], [1, 4, 7, 11]],
[[3, 7, 10, 2], [5, 8, 0, 3], [7, 10, 2, 5], [8, 0, 3, 7], [10, 2, 5, 8], [0, 3, 7, 10], [2, 5, 8, 0]],
[[4, 8, 11, 3], [6, 9, 1, 4], [8, 11, 3, 6], [9, 1, 4, 8], [11, 3, 6, 9], [1, 4, 8, 11], [3, 6, 9, 1]],
[[5, 9, 0, 4], [7, 10, 2, 5], [9, 0, 4, 7], [10, 2, 5, 9], [0, 4, 7, 10], [2, 5, 9, 0], [4, 7, 10, 2]],
[[6, 10, 1, 5], [8, 11, 3, 6], [10, 1, 5, 8], [11, 3, 6, 10], [1, 5, 8, 11], [3, 6, 10, 1], [5, 8, 11, 3]],
[[7, 11, 2, 6], [9, 0, 4, 7], [11, 2, 6, 9], [0, 4, 7, 11], [2, 6, 9, 0], [4, 7, 11, 2], [6, 9, 0, 4]],
[[8, 0, 3, 7], [10, 1, 5, 8], [0, 3, 7, 10], [1, 5, 8, 0], [3, 7, 10, 1], [5, 8, 0, 3], [7, 10, 1, 5]],
[[9, 1, 4, 8], [11, 2, 6, 9], [1, 4, 8, 11], [2, 6, 9, 1], [4, 8, 11, 2], [6, 9, 1, 4], [8, 11, 2, 6]],
[[10, 2, 5, 9], [0, 3, 7, 10], [2, 5, 9, 0], [3, 7, 10, 2], [5, 9, 0, 3], [7, 10, 2, 5], [9, 0, 3, 7]],
[[11, 3, 6, 10], [1, 4, 8, 11], [3, 6, 10, 1], [4, 8, 11, 3], [6, 10, 1, 4], [8, 11, 3, 6], [10, 1, 4, 8]],
[[0, 4, 7, 11], [2, 5, 9, 0], [4, 7, 11, 2], [5, 9, 0, 4], [7, 11, 2, 5], [9, 0, 4, 7], [11, 2, 5, 9]],
[[1, 4, 8, 11], [3, 6, 9, 1], [4, 8, 11, 3], [6, 9, 1, 4], [8, 11, 3, 6], [9, 1, 4, 8], [11, 3, 6, 9]],
[[2, 5, 9, 0], [4, 7, 10, 2], [5, 9, 0, 4], [7, 10, 2, 5], [9, 0, 4, 7], [10, 2, 5, 9], [0, 4, 7, 10]],
[[3, 6, 10, 1], [5, 8, 11, 3], [6, 10, 1, 5], [8, 11, 3, 6], [10, 1, 5, 8], [11, 3, 6, 10], [1, 5, 8, 11]],
[[4, 7, 11, 2], [6, 9, 0, 4], [7, 11, 2, 6], [9, 0, 4, 7], [11, 2, 6, 9], [0, 4, 7, 11], [2, 6, 9, 0]],
[[5, 8, 0, 3], [7, 10, 1, 5], [8, 0, 3, 7], [10, 1, 5, 8], [0, 3, 7, 10], [1, 5, 8, 0], [3, 7, 10, 1]],
[[6, 9, 1, 4], [8, 11, 2, 6], [9, 1, 4, 8], [11, 2, 6, 9], [1, 4, 8, 11], [2, 6, 9, 1], [4, 8, 11, 2]],
[[7, 10, 2, 5], [9, 0, 3, 7], [10, 2, 5, 9], [0, 3, 7, 10], [2, 5, 9, 0], [3, 7, 10, 2], [5, 9, 0, 3]],
[[8, 11, 3, 6], [10, 1, 4, 8], [11, 3, 6, 10], [1, 4, 8, 11], [3, 6, 10, 1], [4, 8, 11, 3], [6, 10, 1, 4]],
[[9, 0, 4, 7], [11, 2, 5, 9], [0, 4, 7, 11], [2, 5, 9, 0], [4, 7, 11, 2], [5, 9, 0, 4], [7, 11, 2, 5]],
[[10, 1, 5, 8], [0, 3, 6, 10], [1, 5, 8, 0], [3, 6, 10, 1], [5, 8, 0, 3], [6, 10, 1, 5], [8, 0, 3, 6]],
[[11, 2, 6, 9], [1, 4, 7, 11], [2, 6, 9, 1], [4, 7, 11, 2], [6, 9, 1, 4], [7, 11, 2, 6], [9, 1, 4, 7]],]
class DataManager:
    file_names:     [str]
    key_order:      [str]
    settings:       Data
    params:         Params

    def __init__(self, params: Params, settings: Data):
        self.params         = params
        self.settings       = settings
        self.files          = glob.glob(str(pathlib.Path(self.settings.input)/"**/*.mid*"))
        self.file_names     = self.files[self.settings.offset:self.settings.offset+self.settings.amount]
        self.key_order      = ["pitch", "step", "duration"]

    def generate_dataset(self):
        data = None
        data_length = 0
        offset = self.settings.offset
        file_count = self.settings.amount
        for file in self.file_names:
            notes = midi_to_notes(file, self.params.steps_per_second)
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
        if not os.path.exists(self.settings.dataset_dir):
            os.mkdir(self.settings.dataset_dir)
        path = f"{self.settings.dataset_dir}{self.settings.name}"
        training_data.save(f"{self.settings.dataset_dir}{self.settings.name}")
        
        with open( f"{path}/settings.json", "w") as f:
            f.write(json.dumps({
                "nudge_step":       self.settings.nudge_step,
                "nudge_duration":   self.settings.nudge_duration,
                "offset":           self.settings.offset,
                "amount":           self.settings.amount,
                "dataset_path":     self.settings.input,
                "total_file_count": len(self.files)
            }))


    def create_sequences(self, dataset: tf.data.Dataset, file_path: str):
        """Returns TF Dataset of sequence and label examples."""
        sequence_length = 64+1

        # Take 1 extra for the labels
        windows = dataset.window(sequence_length, shift=1, stride=1,
                                    drop_remainder=True)

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(sequence_length, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        

        # Normalize note pitch
        def scale_pitch(x):
            x = x/[self.params.vocab_size,1.0,1.0]
            #p, s, d = tf.split(x, num_or_size_splits=3, axis=-1)
            #return (p, s, d)
            return x

        def analyze_sequence(sequence):
            step = tf.slice(sequence, [self.params.sequence_length - 10, 1], [10, 1])
            dur = tf.slice(sequence, [self.params.sequence_length - 32, 2], [32, 1])
            max = tf.reduce_max(dur)
            min = tf.reduce_min(dur)
            test = tf.less_equal(step, tf.constant([0.1], dtype=tf.float64))
            r = tf.reduce_any(test, 1)
            sum = tf.reduce_sum(tf.cast(r, tf.float32))
            return sum, max, min
            
        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            labels_dense = sequences[-1]
            labels = {key:labels_dense[i] for i,key in enumerate(self.key_order)}
            return scale_pitch(inputs), labels, self.get_key_in_filename(file_path), analyze_sequence(sequences)

        s = sequences.map(split_labels)
        return s

    def anders_pre(self, key, notes):
        acceptable_chords = chords[key]
        notes = np.flip(np.mod(notes, 12)) # Why do we flip it
        losses = np.full(12, loss_const)
        key_notes = KEYS[key].numpy()

    
    def get_average_chord_length(self, key, notes, acceptable_chords):
        notes_len = len(notes)
        chords = 0
        chords_count = 0
        failed_notes = 0
        for i in range(notes_len):
            for idx, chord in enumerate(acceptable_chords):
                for j in range(i, notes_len):
                    match = False
                    for k in range(0, 4):
                        if notes[j] == chord[k]:
                            i += 1
                            match = True
                    if not match:
                        failed_notes += 1
                        break
                    else:
                        chords_count +=1
            # Entry point
            if chords_count > 0:
                chords += 1
            chords_count = 0
        return (notes_len - failed_notes) / chords



    def get_key_in_filename(self, file_path: str):
        with open("./keys.json") as file:
            jsonObject = json.load(file)
            file.close()

        for x in jsonObject:
            if (x["file_path"][5:-1] == file_path[8:-1]):
                return x["key"]


