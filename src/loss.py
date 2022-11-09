from consts import *
import numpy as np

class MusicLoss():
    key_weight: float
    batch_size: int
    def __init__(self, batch_size, key_weight = 0.5):
        self.key_weight = key_weight
        self.batch_size = batch_size
    
    @tf.function
    def __call__(self, y_true, y_pred, keys: tf.Tensor):
        index = tf.random.categorical(y_pred, num_samples=1)
        y = tf.math.floormod(index, 12)
        equal = tf.math.equal(y, keys)
        f = tf.reduce_any(equal, 1)
        tr = tf.fill((self.batch_size, 1), 1.0)
        return sparse_entrophy(y_true, y_pred) + self.key_weight * cross_entropy(f, tr)
    
    def preprocessing(self, key, song_notes):

        acceptable_chords = initialize_acceptable_chord_set(key)
        song_notes = normalize_song_notes(song_notes)
        print("normalized song notes: ", song_notes)
        average_chord_length = get_average_chord_length(key, song_notes, acceptable_chords)
        current_possible_chords = get_possible_chords(key, song_notes, acceptable_chords)
        #get set of chords in current key
        output_losses = [1,1,1,1,1,1,1,1,1,1,1,1]
        output_losses = list(map(lambda n: n * loss_constant, output_losses))
        print("first output losses: ", output_losses)
        notes_played_in_latest_cord = 0
        #count the latest notes played in latest chord
        for x in current_possible_chords:
            if(x>notes_played_in_latest_cord):
                notes_played_in_latest_cord=x
        print("notes played in latest chord: ", notes_played_in_latest_cord)
        if(notes_played_in_latest_cord<average_chord_length):
            for idx, y in enumerate(current_possible_chords):
                if(y==notes_played_in_latest_cord):
                    for z in acceptable_chords[idx]:
                        output_losses[z]=0
            for note in output_losses:
                if(note != 0):
                    note = 2*loss_constant
        key_notes = KEYS[key].numpy()
        print("key notes: ", key_notes)
        #assign highest
        for idx in range(0,12):
            if idx not in key_notes:
                output_losses[idx]=3*loss_constant
        print("output losses: ", output_losses)
        return output_losses


def initialize_acceptable_chord_set(key):
    #get set of chords in current key
    if(key < 12):
        acceptable_chords = major_chords
    else:
        acceptable_chords = minor_chords
    #map chords so 0 is at our specific key note
    counter = 0
    distance_from_C = key % 12
    while(counter < 7):
        acceptable_chords[counter] = list(map(lambda n: (n + distance_from_C) % 12, acceptable_chords[counter]))
        counter +=1
    print("first acceptable chord: ", list(acceptable_chords[0]))
    print("second acceptable chord: ", list(acceptable_chords[1]))
    print("last acceptable chord: ", list(acceptable_chords[6]))
    return acceptable_chords
            
def normalize_song_notes(song_notes): #normalizes notes to exclude octave and flips the array to fit the function design
    for note in song_notes:
        note %= 12
    return np.flip(song_notes)

def get_average_chord_length(key, song_notes, acceptable_chords):
    chord_index_counter = [0, 0, 0, 0, 0, 0, 0]
    song_note_counter = 0
    song_chord_counter = 0
    song_length = len(song_notes)

    failed_notes = 0
    while(song_note_counter < song_length):
        for idx, chord in enumerate(acceptable_chords):
            for song_note in range(song_note_counter, song_length):
                #print(song_note)
                note_belongs = False
                for k in range(0, 4):
                    if(chord[k] == song_notes[song_note]):
                        chord_index_counter[idx] += 1
                        note_belongs = True
                if( note_belongs==False):
                    break
            
        
        longest_chord = 0
        for i in range(0, 7):
            if(chord_index_counter[i]>longest_chord):
                longest_chord = chord_index_counter[i]
            chord_index_counter[i] = 0
        if(longest_chord == 0):
            failed_notes += 1#make counter of first and last failed note, to check if they only appear in c piece
            song_note_counter += 1
        else:
            song_note_counter += longest_chord
            song_chord_counter += 1
    average_chord_length = (song_length- failed_notes)/song_chord_counter
    print("failed notes: ", failed_notes)
    print("average chord length: ", average_chord_length)
    return average_chord_length


def get_possible_chords(key, song_notes, acceptable_chords):
    print("song notes in possible chords: ", song_notes)
    acceptable_chord_number = [0,0,0,0,0,0,0]

    for idx, chord in enumerate(acceptable_chords):
        print("chord: ", chord)
        for prev_note in song_notes:
            note_belongs = False
            for note in chord:
                if(note == prev_note):
                    acceptable_chord_number[idx] += 1
                    note_belongs = True
            if(note_belongs==False):
                break
    print("acceptable chord numbers", acceptable_chord_number)
    return acceptable_chord_number