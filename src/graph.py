import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import json
from dataclasses import dataclass, asdict
from simple_parsing import ArgumentParser
# Used to convert old format into the new
def fix_json(file: str):
    res = {}
    with open(file) as f:
        data = json.load(f)
        data["training_loss"] = convert(data, "training_loss")
        data["test_loss"] = convert(data, "test_loss")
        res = data

    with open(file, "w") as f:
        f.write(json.dumps(res))

def convert(data: dict, t: str):
    loss = []
    pitch = []
    duration = []
    step = []
    for epoch in data[t]:
        loss.append(data[t][epoch]["loss"])
        pitch.append(data[t][epoch]["pitch"])
        duration.append(data[t][epoch]["duration"])
        step.append(data[t][epoch]["step"])
    return {"loss": loss, "pitch": pitch, "duration": duration, "step": step}



def load_data(file: str) -> dict:
    with open(file) as f:
        return json.load(f)



def line_plot(plot: plt.Axes, numbers: [int], title: str, ylabel: str = "", xlabel: str = "Epochs"):
    plot.plot(numbers)
    plot.set_title(title)
    plot.set_ylabel(ylabel)
    plot.set_xlabel(xlabel)

def plot_loss(data: dict, data_type: str, loss: bool, pitch: bool, step: bool, duration: bool):
    left = int(loss) + int(pitch)
    right = int(step) + int(duration)
    figure_count = 1
    if left > 0 and right > 0:
        figure_count = 2
    if left + right == 0:
        raise Exception("Cannot generate plot when no loss type is enabled")
    
    figure = plt.figure(constrained_layout=True)
    figure.suptitle(f"The evolution of loss during {data_type}", fontsize="xx-large")
    subfigures = figure.subfigures(1, figure_count, wspace=0.02)
    figure_curr = subfigures
    if left > 0:
        count = 1
        if loss and pitch:
            count = 2
        if right > 0: figure_curr = subfigures[0]
        axs_left = figure_curr.subplots(count, 1, sharex=True)
        if issubclass(type(axs_left), Axes):
            axs_left = [axs_left, axs_left]
        if loss:
            line_plot(axs_left[0], data["loss"],    "Total loss")
        if pitch:
            line_plot(axs_left[1], data["pitch"],   "Pitch loss")
    
    if right > 0:
        count = 1
        if step and duration:
            count = 2
        if right > 0: figure_curr = subfigures[1]
        axs_right = figure_curr.subplots(count, 1, sharex=True)
        if issubclass(type(axs_right), Axes):
            axs_right = [axs_right, axs_right]
        if step:
            line_plot(axs_right[0], data["step"],     "Step loss")
        if duration:
            line_plot(axs_right[1], data["duration"], "Duration loss")
    return figure


@dataclass
class Params:
    """ Parameters for generating loss graphs for the model """
    model:      str                     # Sets the model to generate graphs from.
    models_dir: str     = "./models"    # Sets the models directory.
    train:      bool    = True          # Enables graphing for train data.
    test:       bool    = True          # Enables graphing for test data.
    loss:       bool    = True          # Enables loss graph.
    pitch:      bool    = True          # Enables pitch loss graph.
    step:       bool    = True          # Enables step loss graph.
    duration:   bool    = True          # Enables duration loss graph.
    save:       bool    = False         # Instead of showing the graph it will be saved as an svg.

parser = ArgumentParser(
    prog = "Graph from model",
    description = "Generates a series of graphs for the training and testing loss",
    epilog = "Oh lord what are you looking for ;-;"
)
parser.add_arguments(Params, dest="params")

def main():
    args = parser.parse_args()
    params: Params = args.params
    data = load_data(f"./{params.models_dir}/{params.model}/{params.model}.json")

    if not params.test and not params.train:
        raise Exception("No graphs to display when both training and testing graphs are disabled")
    if params.test:
        figure = plot_loss(data["test_loss"], "testing", params.loss, params.pitch, params.step, params.duration)
        if params.save:
            figure.savefig(f"./{params.models_dir}/{params.model}/testing.svg", format="svg")
    if params.train:
        figure = plot_loss(data["training_loss"], "training", params.loss, params.pitch, params.step, params.duration)
        if params.save:
            figure.savefig(f"./{params.models_dir}/{params.model}/training.svg", format="svg")
    if not params.save:
        plt.show()

main()
