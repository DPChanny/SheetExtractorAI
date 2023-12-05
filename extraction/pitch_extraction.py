import math
from typing import TYPE_CHECKING

from numpy import array

if TYPE_CHECKING:
    from extraction.feature_extraction import STFTFeature
    from extraction.beat_extraction import Beat
    from public import Sample

pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
pitch_octave_names = []


def extract_beat_frequencies(sample: "Sample",
                             stft_feature: "STFTFeature",
                             beats: list["Beat"],
                             log: bool = False) -> list[float]:
    if log:
        print("Extracting " + sample.sample_name + " beat frequencies")
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


def extract_beat_pitches(sample: "Sample",
                         frequencies: list[float],
                         max_octave: int = 8,
                         log: bool = False) -> list[str]:
    if log:
        print("Extracting " + sample.sample_name + " beat pitches")
    beat_pitches = []
    for frequency in frequencies:
        beat_pitches.append(frequency_to_pitch(frequency, max_octave))
    return beat_pitches
