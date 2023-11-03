import math


pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
pitch_octave_names = []


def get_pitch_octave_names(max_octave):
    pitch_octave_names.clear()
    for octave in range(0, max_octave):
        for pitch_name in pitch_names:
            pitch_octave_names.append(pitch_name + str(octave + 1))


def frequency_to_pitch(frequency, max_octave):
    get_pitch_octave_names(max_octave)
    return pitch_octave_names[pitch_octave_names.index("A4") + round(math.log(frequency / 440, 2 ** (1 / 12)))]


def extract_pitches(frequencies, max_octave):
    pitches = []
    for frequency in frequencies:
        pitches.append(frequency_to_pitch(frequency, max_octave))
    print(pitches)
    return pitches
