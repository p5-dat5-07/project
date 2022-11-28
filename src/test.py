from consts import *
import numpy as np
import json
loss_const = 1
a_chords = [[[0, 4, 7, 11], [2, 5, 9, 0], [4, 7, 11, 2], [5, 9, 0, 4], [7, 11, 2, 5], [9, 0, 4, 7], [11, 2, 5, 9]],
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

def anders_pre(key, notes):
    acceptable_chords = a_chords[key]
    notes = np.flip(np.mod(notes, 12)) # Why do we flip it
    losses = np.full(12, loss_const)
    key_notes = KEYS[key].numpy()

    achord_length = get_average_chord_length(key, notes, acceptable_chords)
    chords = get_possible_chords(key, notes, acceptable_chords)

    longest_chord = np.argmax(chords)

    if(chords[longest_chord] > achord_length):
        for z in acceptable_chords[longest_chord]:
            losses[z]=0
        for note in losses:
            if(note != 0):
                note = 2*loss_const
    for idx in range(0,12):
        if idx not in key_notes:
            losses[idx]=3*loss_const
    
    return losses

def get_average_chord_length(key, notes, acceptable_chords):
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

def get_possible_chords(key, song_notes, acceptable_chords):
    acceptable_chord_number = np.full(7, 0)

    for idx, chord in enumerate(acceptable_chords):
        for prev_note in song_notes:
            match = False
            for note in chord:
                if(note == prev_note):
                    acceptable_chord_number[idx] += 1
                    match = True
            if(match==False):
                break
    return acceptable_chord_number 


with open("./src/trainingData.json") as f:
    data = json.load(f)
    i = 0
    for test in data:
        res = anders_pre(test['key'], test['sequence'])
        i+=1
        if (res != test['output']).all():
            print(f"Test #{i}: Failed got {res} expected {test['output']}")
        else:
            print(f"Test #{i}: Success")
#print(anders_pre)