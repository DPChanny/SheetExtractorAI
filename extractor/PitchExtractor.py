import math

from numpy import array

import Sample
from Public import STFTFeature, Beat

pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
pitch_octave_names = []


def extract_beat_frequencies(sample: Sample, stft_feature: STFTFeature, beats: list[Beat]) -> list[float]:
    beat_frequencies = []
    for index, beat in enumerate(beats):
        frequencies = array([sum(frequencies[beat.start:beat.end]) for frequencies in stft_feature.magnitudes_db])
        beat_frequencies.append(frequencies.argmax() / len(stft_feature.magnitudes_db) * sample.sampling_rate / 2)
    return beat_frequencies


def get_pitch_octave_names(max_octave: int = 8):
    pitch_octave_names.clear()
    for octave in range(0, max_octave):
        for pitch_name in pitch_names:
            pitch_octave_names.append(pitch_name + str(octave + 1))


def frequency_to_pitch(frequency, max_octave: int = 8):
    get_pitch_octave_names(max_octave)
    return pitch_octave_names[pitch_octave_names.index("A4") + round(math.log(frequency / 440, 2 ** (1 / 12)))]


def extract_beat_pitches(sample: Sample,
                         stft_feature: STFTFeature,
                         beats: list[Beat],
                         max_octave: int = 8) -> list[str]:
    beat_pitches = []
    for frequency in extract_beat_frequencies(sample, stft_feature, beats):
        beat_pitches.append(frequency_to_pitch(frequency, max_octave))
    return beat_pitches
