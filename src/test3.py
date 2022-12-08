start = 505
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
sample_dir = "./data/q-maestro-v2.0.0"
other_dir = "./final_models_with_pinao_roll"
files = glob.glob(str(pathlib.Path(sample_dir)/"**/*.mid*"))
sample_files = glob.glob(str(pathlib.Path(other_dir)/"**/**/*.mid*"))
import pretty_midi 
import collections
import math

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

base = []
for i in range(start, start+sample_count):
    key = get_key_in_filename(files[i])
    base.append(key)
    print(f"{files[i]} {i}: {key}")

res = []
for file in sample_files:
    key = base[int(file[-5:-4])-1]
    notes = midi_to_notes(file, 1)
    fail, ok = in_key(notes["pitch"][64:], key)
    leaps, average = _leaps(notes["pitch"][63:])
    length = len(notes["pitch"][64:])
    res.append({"name": file, "notes": length, "key": key, "fail": fail, "ok": ok, "ok ratio": ok/length, "fail ratio": fail/length,  "leaps": leaps, "average leap": average})

with open("./res2.json", "w") as f:
    f.write(json.dumps(res, indent=4))


