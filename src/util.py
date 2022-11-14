import pandas as pd
import pretty_midi 

def midi_to_notes(self, steps_per_seconds: int, file: str) -> pd.DataFrame:
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
        notes["step"].append((start - prev_start) * steps_per_seconds)
        notes["duration"].append((end - start) * steps_per_seconds)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(notes: pd.DataFrame, instrument_name: str, steps_per_second: int, velocity: int = 100) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + (note["step"]  / steps_per_seconds))
        end = float(start + (note["duration"] / steps_per_seconds))
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
