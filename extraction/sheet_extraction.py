from os import makedirs
from os.path import join
from typing import TYPE_CHECKING

from music21.metadata import Metadata
from music21.note import Note, Rest
from music21.pitch import Pitch
from music21.stream import Stream

if TYPE_CHECKING:
    from extraction.beat_extraction import Beat
    from public import Sample


def extract_sheet(sample: "Sample", beats: list["Beat"], beat_pitches: list[str], log: bool = False) -> Stream:
    if log:
        print("Extracting " + sample.sample_name + " sheet")
    file_stream = Stream()
    file_metadata = Metadata()
    file_metadata.title = sample.sample_name
    file_metadata.composer = "Sheet Extractor AI"
    file_stream.metadata = file_metadata
    for index, beat in enumerate(beats):
        if beat.note:
            note = Note()
            note.pitch = Pitch(beat_pitches[index])
            note.duration.type = beat.beat_type.value
            file_stream.append(note)
        else:
            rest = Rest()
            file_stream.append(rest)
    return file_stream


def save_sheet(sheet: Stream, directory: list[str], sheet_name: str, log: bool = False):
    directory = join(*directory)
    if log:
        print("Saving " + join(*[directory, sheet_name + ".xml"]))
    makedirs(directory, exist_ok=True)
    sheet.write(fp=join(*[directory, sheet_name + ".xml"]))
