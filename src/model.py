import tensorflow as tf
import pandas as pd
import numpy as np
import pretty_midi
import glob
import pathlib
import json
import os
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM, Bidirectional
from callback import Callback
from consts import *
from data_manager import DataManager
from args import Params, Train
from util import midi_to_notes, notes_to_midi

class Model:
    params:         Params
    model:          tf.keras.Model  
    pitch_loss:     Loss
    step_loss:      Loss
    duration_loss:  Loss
    optimizer:      Optimizer
    key_order:      [str]
    training_data:  tf.data.Dataset

    def __init__(self, params: Params, pitch_loss: Loss, step_loss: Loss, duration_loss: Loss,
                    optimizer: Optimizer):
        self.params         = params
        self.pitch_loss     = pitch_loss
        self.step_loss      = step_loss
        self.duration_loss  = duration_loss
        self.optimizer      = optimizer
        self.key_order      = ["pitch", "step", "duration"]
        
    def load(self, model_name):
        self.model = tf.keras.models.load_model(f"./models/{model_name}/{model_name}.h5")
    
    def load_dataset(self, data_path: str, sample_dir: str):
        self.training_data = tf.data.Dataset.load(data_path)
        with open(f"{data_path}/settings.json", "r") as f:
            settings = json.load(f)
            if sample_dir == settings['dataset_path']:
                if self.params.sample_location + self.params.samples_per_epoch > settings['total_file_count']:
                    print(f"Sample location ({self.params.sample_location}) + samples per epoch ({self.params.samples_per_epoch}) has to be lower than the total amount of files ({settings['total_file_count']})!")
                    exit()
                elif max(0, min(self.params.samples_per_epoch+self.params.sample_location, settings["amount"]+settings["offset"]) - max(self.params.sample_location, settings["offset"])+1):
                    print(f"Sample area overlaps with training data!")
                    print(f"\tSample area: {self.params.sample_location}-{self.params.sample_location +  self.params.samples_per_epoch}")
                    print(f"\tTraining area: {settings['offset']}-{settings['offset'] +  settings['amount']}")
                    exit()

    def summary(self):
        print(self.params.summary())
        self.model.summary()
    
    def create_model(self):
        input_shape = (self.params.sequence_length, 3)  
        input_layer = tf.keras.Input(input_shape)

        x = Bidirectional(LSTM(256, return_sequences=True))(input_layer)
        d1 = Dropout(0.3)(x)

        ph1 = Bidirectional(LSTM(128, name="pitch_hidden1"))(d1)
        sh1 = Bidirectional(LSTM(128, name="step_hidden1"))(d1)
        dh1 = Bidirectional(LSTM(128, name="duration_hidden1"))(d1)

        d2 = Dropout(0.3)(ph1)
        d3 = Dropout(0.3)(sh1)
        d4 = Dropout(0.3)(dh1)

        ph2 = Dense(128,  activation="tanh", name="pitch_hidden2")(d2)
        sh2 = Dense(30,   activation="relu", name="step_hidden2")(d3)
        dh2 = Dense(30,   activation="relu", name="duration_hidden2")(d4)

        d5 = Dropout(0.3)(ph2)
        d6 = Dropout(0.3)(sh2)
        d7 = Dropout(0.3)(dh2)

        output_layers = {
            "pitch": Dense(128, name="pitch")(d5),
            "step": Dense(1,  activation="relu", name="step")(d6),
            "duration": Dense(1, activation="relu", name="duration")(d7),
        }
        self.model = tf.keras.Model(input_layer, output_layers)

    def train_model(self, model_name: str, sample_dir: str, save: bool, callback: Callback) -> tf.data.Dataset:
        if save:
            if not os.path.exists("./models"):
                os.mkdir("./models")

            if os.path.exists(f"./models/{model_name}"):
                print(f"Model with the name {model_name} already exists!")
                exit()
            
            os.mkdir(f"./models/{model_name}")
            self.files = glob.glob(str(pathlib.Path(sample_dir)/"**/*.mid*"))
        # Train on data
        for epoch in range(1, self.params.epochs+1):
            print("epoch:", epoch)
            self.train(self.training_data, callback)
            if save and epoch % (self.params.epochs_between_samples) == 0 and epoch > 0:
                os.mkdir(f"./models/{model_name}/epoch{epoch}")
                for i in range(1, self.params.samples_per_epoch+1):
                    print(f"generating sample: {i}")
                    self.generate_notes(self.files[self.params.sample_location+i], f"./models/{model_name}/epoch{epoch}/{model_name}_{epoch}_{i}.mid")
        
        self.model.save( f"./models/{model_name}/{model_name}.h5")
        with open( f"./models/{model_name}/{model_name}.json", "w") as f:
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

        raw_notes = midi_to_notes(in_file, self.params.steps_per_second)
        generated_notes = {
            "pitch": [],
            "step":  [],
            "duration": [],
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

            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(tf.divide(input_note, [128, 1, 1]), 0), axis=0)
            prev_start = start
        pm = notes_to_midi(pd.DataFrame(generated_notes, columns=(*(self.key_order), "start", "end")), instrument_name, self.params.steps_per_second)
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
    

    
