import numpy as np


def pitch2midi(pitchArray, quantizePitch=False):
    # converts array of Hz to array of Midi numbers
    midi = []
    for pitch in pitchArray:
        if pitch == 0:
            midi.append(0.001)
        else:
            midi.append(singlepitch2midi(pitch, quantizePitch))
    return midi


def singlepitch2midi(pitch, quantizePitch=False):
    # converts Hz value to Midi number
    if quantizePitch:
        midi = np.int(np.round(69 + 12 * np.math.log(pitch / 440.0, 2), 0))
    else:
        midi = 69 + 12 * np.math.log(pitch / 440.0, 2)

    return midi


def midi2pitch(midi):
    return 2**((midi-69)/12.0)*440


def midi2chroma(midi):
    print ("Midi: ", midi)
    if midi==np.float32("NaN"):
        return np.float32("NaN")
    else:
        return int((midi-33)%12)


def pitch2chroma(pitch):
    # converts Hz value to a pitch class between 0 and 11 (A to G#)
    midi = pitch2midi([pitch], quantizePitch=True)[0]
    chroma = int((midi-33)%12)
    return chroma

def get_midi_freq_values():
    # returns all the frequency values corresponding to midi values
    midis = list(range(1, 126))
    freqs = []
    for midi in midis:
        freqs.append(round(midi2pitch(midi), 2))

    return midis, freqs

def midi2note (midi_number):
    # converts midi number to note name
    pitch_classes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

    octave = str(int(np.floor(midi_number / 12) - 1))
    pitch_class = pitch_classes[int(midi_number % 12 - 9)]
    note = pitch_class+octave

    return note

