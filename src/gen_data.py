start = 506
sample_count = 5

KEYS = [
    # Major Keys
    [0, 2, 4, 5, 7, 9, 11],
    [1, 3, 5, 6, 8, 10, 0],
    [2, 4, 6, 7, 9, 11, 1],
    [3, 5, 7, 8, 10, 0, 2],
    [4, 6, 8, 9, 11, 1, 3],
    [5, 7, 9, 10, 0, 2, 4],
    [6, 8, 10, 11, 1, 3, 5],
    [7, 9, 11, 0, 2, 4, 6],
    [8, 10, 0, 1, 3, 5, 7],
    [9, 11, 1, 2, 4, 6, 8],
    [10, 0, 2, 3, 5, 7, 9],
    [11, 1, 3, 4, 6, 8, 10],
    # Minor Keys
    [0, 2, 3, 5, 7, 8, 10],
    [1, 3, 4, 6, 8, 9, 11],
    [2, 4, 5, 7, 9, 10, 0],
    [3, 5, 6, 8, 10, 11, 1],
    [4, 6, 7, 9, 11, 0, 2],
    [5, 7, 8, 10, 0, 1, 3],
    [6, 8, 9, 11, 1, 2, 4],
    [7, 9, 10, 0, 2, 3, 5],
    [8, 10, 11, 1, 3, 4, 6],
    [9, 11, 0, 2, 4, 5, 7],
    [10, 0, 1, 3, 5, 6, 8],
    [11, 1, 2, 4, 6, 7, 9],
]

import glob
import pathlib
import json
import util
model = "large-advanced"
sample_dir = "./data/q-maestro-v2.0.0"
other_dir = f"./final_models4/{model}/epoch20"
files = glob.glob(str(pathlib.Path(sample_dir)/"**/*.mid*"))
sample_files = glob.glob(str(pathlib.Path(other_dir)/"*.mid*"))
import pretty_midi 
import collections
import math
import numpy as np

def midi_to_notes(file: str, steps_per_second: int):
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
        notes["step"].append((start - prev_start) * steps_per_second)
        notes["duration"].append((end - start) * steps_per_second)
        prev_start = start

    return notes

def get_key_in_filename(file_path: str):
    with open("./keys.json") as file:
        jsonObject = json.load(file)
        file.close()

    for x in jsonObject:
        if (x["file_path"][5:-1] == file_path[8:-1]):
            return x["key"]

def in_key(notes, key):
    ok = 0 
    fail = 0
    for note in notes:
        if note % 12 in KEYS[key]:
            ok += 1
        else:
            fail += 1
    return (fail, ok)

def _leaps(notes):
    leap = []
    all = []
    for i in range(1, len(notes)):
        prev = math.floor(notes[i-1] / 12)
        curr = math.floor(notes[i] / 12)
        diff = abs(prev-curr)
        all.append(diff)
        if diff > 1:
            leap.append(diff)
    return len(leap), sum(all) / len(all)

def pitch_test(pitches):
    res = 0
    for pitch in pitches:
        if pitch < 21 or pitch > 108:
            res += 1
    return res

base = []
src_notes = []
for i in range(start, start+sample_count):
    key = get_key_in_filename(files[i])
    base.append(key)
    notes = midi_to_notes(files[i], 1)
    src_notes.append(notes["pitch"][:64])
    print(f"{files[i]} {i}: {key}")
def sort(x):
    score = 0
    if "large" in x:
        score += 10
    score += len(x)
    return score

sample_files.sort(key = lambda x: sort(x))
note_counts = [64, 128]
i = 0
for note_count in note_counts:
    print(note_count)
    res = []
    for file in sample_files:
        key = base[i % 5]
        notes = midi_to_notes(file, 1)
        cut_notes = notes["pitch"][64:64+note_count]
        #if file == "final_models4\\large-advanced\\epoch20\\large-advanced_20_5.mid":
        #    largest = 0
        #    res_key = 0
        #    for i in range(0, 24):
        #        _, ok = in_key(cut_notes, i)
        #        if ok > largest:
        #            largest = ok
        #            res_key = i
        #    print(f"ratio: {largest/len(cut_notes)} key: {key}")


        fail, ok = in_key(cut_notes, key)
        leaps, average = _leaps(cut_notes)
        ifail, iok = in_key(src_notes[i % 5], key)
        ileaps, iaverage = _leaps(src_notes[i % 5])
        length = len(cut_notes)
        invalid_duration = abs(len(notes["pitch"][64:]) - 200)
        res.append({
            "name": file, 
            "notes": length,
            "key": key, 
            "fail": fail,
            "ok": ok,
            "ok ratio": ok/length,
            "fail ratio": fail/length,
            "leaps": leaps,
            "average leap": average,
            "invalid druration": invalid_duration,
            "invalid pitch": pitch_test(cut_notes),
            "src fail": ifail,
            "src ok": iok,
            "src ok ratio": iok/64,
            "src fail ratio": ifail/64,
            "src leaps": ileaps,
            "src average leap": iaverage,
        })
        i+=1

    with open(f"./datares/{model}-{note_count}.json", "w") as f:
        f.write(json.dumps(res, indent=4))


#import matplotlib.pyplot as plt
#from matplotlib.axes import Axes
#def line_plot(plot: plt.Axes, numbers: [int], title: str ="", ylabel: str = "", xlabel: str = "Epochs"):
#    plot.plot(numbers)
#    if title != "":
#        plot.set_title(title)
#    plot.set_ylabel(ylabel)
#    plot.set_xlabel(xlabel)
#def ok_graph(ok):
#    figure = plt.figure(constrained_layout=True)
#    axes = figure.subplots(5, 1)
#    line_plot(axes[0], ok[0], xlabel="5 Epochs", title="In key ratio")
#    line_plot(axes[1], ok[1], xlabel="5 Epochs")
#    line_plot(axes[2], ok[2], xlabel="5 Epochs")
#    line_plot(axes[3], ok[3], xlabel="5 Epochs")
#    line_plot(axes[4], ok[4], xlabel="5 Epochs")
#def leaps_graph(leaps):
#    figure = plt.figure(constrained_layout=True)
#    axes = figure.subplots(5, 1)
#    line_plot(axes[0], leaps[0], xlabel="5 Epochs", title="Leaps")
#    line_plot(axes[1], leaps[1], xlabel="5 Epochs")
#    line_plot(axes[2], leaps[2], xlabel="5 Epochs")
#    line_plot(axes[3], leaps[3], xlabel="5 Epochs")
#    line_plot(axes[4], leaps[4], xlabel="5 Epochs")
#ok = [[],[],[],[],[]]
#leaps = [[],[],[],[],[]]
#for i in range(0, len(res)):
#    if i % 20 == 0 and i != 0:
#        #ok_graph(ok)
#        leaps_graph(leaps)
#        ok = [[],[],[],[],[]]
#        leaps = [[],[],[],[],[]]
#    ok[i % 5].append(res[i]["ok ratio"])
#    leaps[i % 5].append(res[i]["average leap"])

#ok_graph(ok)
#leaps_graph(leaps)
#plt.show()

