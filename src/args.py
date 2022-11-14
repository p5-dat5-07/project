from dataclasses import dataclass
from simple_parsing import ArgumentParser, subgroups
from model import Params
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
class Train(Base):
    name: str = "model"                     # The name of the model.
    data: str = "./data/q-maestro-v2.0.0"   # The path to the data directory.
    save: bool = True                       # Wether to save the model

@dataclass
class Mode:
    mode: Base = subgroups(
        {"generate":Generate, "train": Train}, default=Train()
    ) # Select wheter you want to train a new model or generate from a existing model

parser = ArgumentParser(
    prog = "Music Generator",
    description = "The music generator is used to train and generate music",
    epilog = "Oh lord what are you looking for ;-;")
parser.add_arguments(Mode, dest="mode")
parser.add_arguments(Params, dest="params")