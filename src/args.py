from dataclasses import dataclass, asdict
from simple_parsing import ArgumentParser, subgroups
@dataclass
class Base:
    ...

@dataclass
class Generate(Base):
    input:  str             # The input directory or file to get the initial midi sequence(s) from.
    output: str             # The output directory or file to write the midi sequence(s) to.
    amount: int = 1         # The amount of versions generated per input file.
    name:   str = "model"   # The name of the model.


@dataclass
class Data(Base):
    output: str                             # The output directory or file to write the midi sequence(s) to.
    input:  str = "./data/q-maestro-v2.0.0" # The input directory or file to get the initial midi sequence(s) from.
    amount: int = 10                        # The amount of files to add to the dataset.
    offset: int = 0                         # The starting point in the dataset

@dataclass
class Train(Base):
    name: str = "model"                         # The name of the model.
    data: str = "./datasets/maestro_10"         # The path to the data directory.
    sample_dir: str = "./data/q-maestro-v2.0.0" # The sample director to get the sample midi sequence(s) from.
    save: bool = True                           # Wether to save the model

@dataclass
class Mode:
    mode: Base = subgroups(
        {"generate": Generate, "data": Data, "train": Train}, default=Train()
    ) # Select wheter you want to train a new model or generate from a existing model

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
    epochs_between_samples: int   = 10      # The amount of epochs between generating sample midi files.
    samples_per_epoch:      int   = 5       # The amount of midi file samples.
    sample_location:        int   = 505     # The file to start from in the dataset when generating samples.
    notes_per_sample:       int   = 200     # The amount of notes to generate per sample.
    sample_temprature:      int   = 2       # The temprature of the samples.
    steps_per_second:      int   = 1       # The amount of steps per second.
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
epochs between samples:     {self.epochs_between_samples}
samples_per_epoch:          {self.samples_per_epoch}
sample_location:            {self.sample_location}
notes_per_sample:           {self.notes_per_sample}
sample_temprature:          {self.sample_temprature}
steps per second:           {self.steps_per_second}
normalization:              {self.normalization}
        """
    def to_dict(self) -> dict:
        return asdict(self)

parser = ArgumentParser(
    prog = "Music Generator",
    description = "The music generator is used to train and generate music",
    epilog = "Oh lord what are you looking for ;-;")
parser.add_arguments(Mode, dest="mode")
parser.add_arguments(Params, dest="params")