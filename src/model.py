import tensorflow as tf
import pandas as pd
import numpy as np
import pretty_midi
import glob
import pathlib
import collections
import json
import os
from callback import Callback
from dataclasses import dataclass, asdict
from consts import *
from data_manager import DataManager

#(16 * 120 / 60)
@dataclass
class Params:
    """ Parameters for ajusting the model """
    epochs:                 int   = 50      # The amount of Epochs.
    sequence_length:        int   = 64      # The amount of notes per sequence.
    batch_size:             int   = 50      # The batch size.
    learning_rate:          float = 0.001   # The learning rate.
    pitch_loss_scaler:      float = 0.05    # The amount to scale the pitch loss.
    step_loss_scaler:       float = 1.0     # The amount to scale the step loss.
    duration_loss_scaler:   float = 1.0     # The amount to scale the duration loss.
    vocab_size:             int   = 128     # The amount of pitches in midi file DONT CHANGE!
    file_count:             int   = 7       # The amount of files to extract notes from.
    files_offset:           int   = 500     # The file to start from in the dataset.
    epochs_between_samples: int   = 10      # The amount of epochs between generating sample midi files.
    samples_per_epoch:      int   = 5       # The amount of midi file samples.
    sample_location:        int   = 505     # The file to start from in the dataset when generating samples.
    notes_per_sample:       int   = 200     # The amount of notes to generate per sample.
    sample_temprature:      int   = 2       # The temprature of the samples.
    steps_per_seconds:      int   = 1       # The amount of steps per second.
    normalization:          int   = 0.25    # The normalization value for the gradient.

    def summary(self) -> str:
        return f"""
epochs:                     {self.epochs}
sequence length:            {self.sequence_length}
batch size:                 {self.batch_size}
learning rate:              {self.learning_rate}
pitch loss scaler:          {self.pitch_loss_scaler}
step loss scaler:           {self.step_loss_scaler}
duration loss scaler:       {self.duration_loss_scaler}
vocab size:                 {self.vocab_size}
file count:                 {self.file_count}
files offset:               {self.files_offset}
epochs between samples:     {self.epochs_between_samples}
samples_per_epoch:          {self.samples_per_epoch}
sample_location:            {self.sample_location}
notes_per_sample:           {self.notes_per_sample}
sample_temprature:          {self.sample_temprature}
steps per seconds:          {self.steps_per_seconds}
normalization:              {self.normalization}
        """
    def to_dict(self) -> dict:
        return asdict(self)

class Model:
    params:         Params
    model:          tf.keras.Model  
    pitch_loss:     Loss
    step_loss:      Loss
    duration_loss:  Loss
    optimizer:      Optimizer
    key_order:      [str]

    def __init__(self, params: Params, pitch_loss: Loss, step_loss: Loss, duration_loss: Loss,
                    optimizer: Optimizer):
        self.params         = params
        self.pitch_loss     = pitch_loss
        self.step_loss      = step_loss
        self.duration_loss  = duration_loss
        self.optimizer      = optimizer
        self.key_order      = ["pitch", "step", "duration"]
        self.dm             = DataManager(params)
        
    def load(self, model_name):
        self.model = tf.keras.models.load_model(f"./{model_name}/{model_name}.h5")
    def summary(self):
        print(self.params.summary())
        self.model.summary()
    
    def create_model(self):
        input_shape = (self.params.sequence_length, 3)  
        input_layer = tf.keras.Input(input_shape)

        x = tf.keras.layers.GRU(256, return_sequences=True)(input_layer)
        d1 = tf.keras.layers.Dropout(0.3)(x)

        ph1 = tf.keras.layers.GRU(128, name="pitch_hidden1")(d1)
        sh1 = tf.keras.layers.GRU(128, name="step_hidden1")(d1)
        dh1 = tf.keras.layers.GRU(128, name="duration_hidden1")(d1)

        d2 = tf.keras.layers.Dropout(0.3)(ph1)
        d3 = tf.keras.layers.Dropout(0.3)(sh1)
        d4 = tf.keras.layers.Dropout(0.3)(dh1)

        ph2 = tf.keras.layers.Dense(128,  activation="sigmoid", name="pitch_hidden2")(d2)
        sh2 = tf.keras.layers.Dense(30,   activation="relu", name="step_hidden2")(d3)
        dh2 = tf.keras.layers.Dense(30,   activation="relu", name="duration_hidden2")(d4)

        d5 = tf.keras.layers.Dropout(0.3)(ph2)
        d6 = tf.keras.layers.Dropout(0.3)(sh2)
        d7 = tf.keras.layers.Dropout(0.3)(dh2)

        output_layers = {
            "pitch": tf.keras.layers.Dense(128, name="pitch")(d5),
            "step": tf.keras.layers.Dense(1,  activation="relu", name="step")(d6),
            "duration": tf.keras.layers.Dense(1, activation="relu", name="duration")(d7),
        }
        self.model = tf.keras.Model(input_layer, output_layers)

    def train_model(self, model_name: str, callback: Callback) -> tf.data.Dataset:
        training_data = self.dm.load_dataset()
        
        if os.path.exists(f"./{model_name}"):
            print(f"Model with the name {model_name} already exists!")
            exit()

        # Train on data
        os.mkdir(f"./{model_name}")
        for epoch in range(self.params.epochs):
            print("epoch:", epoch)
            self.train(training_data, callback)
            if epoch % (self.params.epochs_between_samples-1) == 0 and epoch > 0:
                os.mkdir(f"./{model_name}/epoch{epoch}")
                for i in range(0, self.params.samples_per_epoch):
                    print(f"generating sample: {i+1}")
                    self.generate_notes(self.files[self.params.sample_location+i], f"./{model_name}/epoch{epoch}/{model_name}_{epoch}_{i}.mid")
        
        self.model.save( f"./{model_name}/{model_name}.h5")
        with open( f"./{model_name}/{model_name}.json", "w") as f:
            f.write(json.dumps(self.params.to_dict()))
    
    @tf.function
    def train_step(self, x_batch_train, y_batch_train, keys: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = self.model(x_batch_train, training=True)  # Logits for this minibatch
        
            # Compute the loss value for this minibatch.
            pitch_loss = self.pitch_loss(y_batch_train["pitch"], logits["pitch"],  KEYS[keys[-1]]) * self.params.pitch_loss_scaler
            step_loss = self.step_loss(y_batch_train["step"], logits["step"]) * self.params.step_loss_scaler
            duration_loss = self.duration_loss(y_batch_train["duration"], logits["duration"]) * self.params.duration_loss_scaler

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        gradients = tape.gradient([pitch_loss, step_loss, duration_loss], self.model.trainable_variables)
        grads = [tf.clip_by_norm(g, self.params.normalization)
                for g in gradients]

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return (pitch_loss, step_loss, duration_loss)

    def train(self, training_data: tf.data.Dataset, callback: Callback):
        for step, (x_batch_train, y_batch_train, keys) in enumerate(training_data):
            (pitch_loss, step_loss, duration_loss) = self.train_step(x_batch_train, y_batch_train, keys)
            callback(step, pitch_loss, step_loss, duration_loss)

    def generate_notes(self, in_file: str, out_file: str):
        instrument = pretty_midi.PrettyMIDI(in_file).instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

        raw_notes = self.dm.midi_to_notes(in_file)
        generated_notes = {
            "pitch": [],
            "step":  [],
            "duration": [],
            "start": [],
            "end":  []
        }

        sample_notes = np.stack([raw_notes[key] for key in self.key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training
        # sequences
        input_notes = (
            sample_notes[:self.params.sequence_length] / np.array([self.params.vocab_size, 1, 1]))

        prev_start = 0
        # Generates
        for i in range(self.params.notes_per_sample):
            pitch, step, duration = self.predict_next_note(input_notes, self.params.sample_temprature) # temprature param
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)

            generated_notes["pitch"].append(pitch)
            generated_notes["step"].append(step)
            generated_notes["duration"].append(duration)
            generated_notes["start"].append(start)
            generated_notes["end"].append(end)

            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(tf.divide(input_note, [128, 1, 1]), 0), axis=0)
            prev_start = start
        pm = self.notes_to_midi(pd.DataFrame(generated_notes, columns=(*(self.key_order), "start", "end")), instrument_name)
        pm.write(out_file)

    def predict_next_note(
        self,
        notes: np.ndarray, 
        temperature: float = 1.0
        ) -> int:
        """Generates a note IDs using a trained sequence model."""

        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = self.model.predict(inputs, verbose=0)
        pitch_logits = predictions["pitch"]
        step = predictions["step"]
        duration = predictions["duration"]

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)

        # `step` and `duration` values should be non-negative
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration)
    
    def notes_to_midi(self,
        notes: pd.DataFrame,
        instrument_name: str,
        velocity: int = 100,  # note loudness
        ) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program(
                instrument_name))

        prev_start = 0
        for i, note in notes.iterrows():
            start = float(prev_start + (note["step"]  / self.params.steps_per_seconds))
            end = float(start + (note["duration"] / self.params.steps_per_seconds))
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(note["pitch"]),
                start=start,
                end=end,
            )
            instrument.notes.append(note)
            prev_start = start

        pm.instruments.append(instrument)
        return pm
    
