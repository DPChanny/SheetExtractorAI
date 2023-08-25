from FrequencyExtractor import extract_frequencies
from PitchExtractor import extract_pitches
import music21.stream

duration_types = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th", 32: "32nd", 64: "64th"}


def extract_sheet(file_name, bar_count, beat_count_per_bar, max_octave):
    pitches = extract_pitches(extract_frequencies(file_name, bar_count, beat_count_per_bar), max_octave)
    file_stream = music21.stream.Stream()
    file_metadata = music21.metadata.Metadata()
    file_metadata.title = file_name
    file_metadata.composer = "Sheet Extractor"
    file_stream.metadata = file_metadata
    for pitch in pitches:
        note = music21.note.Note(pitch)
        note.duration.type = duration_types[beat_count_per_bar]
        file_stream.append(note)
    file_stream.write(fp=file_name + ".xml")
