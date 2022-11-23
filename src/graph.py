import matplotlib.pyplot as plt
import json
from dataclasses import dataclass, asdict
from simple_parsing import ArgumentParser
# Used to convert old format into the new
def fix_json(file: str):
    res = {}
    with open(file) as f:
        data = json.load(f)
        data['training_loss'] = convert(data, 'training_loss')
        data['test_loss'] = convert(data, 'test_loss')
        res = data

    with open(file, 'w') as f:
        f.write(json.dumps(res))

def convert(data: dict, t: str):
    loss = []
    pitch = []
    duration = []
    step = []
    for epoch in data[t]:
        loss.append(data[t][epoch]['loss'])
        pitch.append(data[t][epoch]['pitch'])
        duration.append(data[t][epoch]['duration'])
        step.append(data[t][epoch]['step'])
    return {"loss": loss, "pitch": pitch, "duration": duration, "step": step}



def load_data(file: str) -> dict:
    with open(file) as f:
        return json.load(f)



def line_plot(plot: plt.Axes, numbers: [int], title: str, ylabel: str, xlabel: str = 'Epochs'):
    x = []
    for i in range(len(numbers)):
        x.append(i)
    plot.plot(numbers)
    plot.set_title(title)
    plot.set_ylabel(ylabel)
    plot.set_xlabel(xlabel)

def plot_loss(data: dict, data_type: str, loss: bool, pitch: bool, step: bool, duration: bool):
    left = int(loss) + int(pitch)
    right = int(step) + int(duration)

    if left + right == 0:
        raise Exception("Cannot generate plot when no loss type is enabled")
    fig, axs = plt.subplots(left, right)
    fig.suptitle(f"The evolution of loss during the {data_type} phase")
    if loss:
        line_plot(axs[0,0], data['loss'],     f"Total loss during {data_type}",           "Total loss")
    if pitch:
        line_plot(axs[1,0], data['pitch'],    f"Total pitch loss during {data_type}",     "Pitch loss")
    if step:
        line_plot(axs[0,1], data['step'],     f"Total step loss during {data_type}",      "Step loss")
    if duration:
        line_plot(axs[1,1], data['duration'], f"Total duration loss during {data_type}",  "Duration loss")
    fig.tight_layout()


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
        plot_loss(data['test_loss'], "testing", params.loss, params.pitch, params.step, params.duration)
    if params.train:
        plot_loss(data['training_loss'], "training", params.loss, params.pitch, params.step, params.duration)
    plt.show()

main()
