from extraction.beat_extraction import (BeatStateExtractor,
                                        extract_beat_data_frame,
                                        save_beat_data_frame,
                                        extract_beats)
from extraction.feature_extraction import (extract_stft_feature,
                                           save_stft_feature_plot,
                                           extract_wave_feature,
                                           save_wave_feature_plot)
from extraction.pitch_extraction import extract_beat_frequencies, extract_beat_pitches
from extraction.sheet_extraction import extract_sheet, save_sheet
from public import Sample

LOG = True

WING_LENGTH = 5

BEAT_STATE_EXTRACTOR_NAME = "beat_state_extractor"

SOURCE = ".\\source"
RESULT = ".\\result"

samples = [Sample([SOURCE, "sample"], "marimba_60_quarter_rest", 60, log=LOG),
           Sample([SOURCE, "sample"], "marimba_60_quarter_note", 60, log=LOG)]

beat_state_extractor = BeatStateExtractor(WING_LENGTH)
beat_state_extractor.load([SOURCE, BEAT_STATE_EXTRACTOR_NAME], BEAT_STATE_EXTRACTOR_NAME, log=LOG)
beat_state_extractor.compile()

for sample in samples:
    wave_feature = extract_wave_feature(sample, log=LOG)
    stft_feature = extract_stft_feature(sample, log=LOG)
    beat_data_frame = extract_beat_data_frame(stft_feature, wing_length=WING_LENGTH, log=LOG)
    beat_states = beat_state_extractor.extract_beat_states(sample, beat_data_frame, log=LOG)
    beats = extract_beats(sample, beat_states, log=LOG)
    beat_frequencies = extract_beat_frequencies(sample, stft_feature, beats, log=LOG)
    beat_pitches = extract_beat_pitches(sample, beat_frequencies, log=LOG)
    sheet = extract_sheet(sample, beats, beat_pitches, log=LOG)

    save_wave_feature_plot(sample,
                           wave_feature,
                           [RESULT, "sample", sample.sample_name],
                           sample.sample_name, log=LOG)
    save_stft_feature_plot(sample,
                           stft_feature,
                           [RESULT, "sample", sample.sample_name],
                           sample.sample_name,
                           beats=beats,
                           beat_states=beat_states,
                           log=LOG)
    save_beat_data_frame(beat_data_frame,
                         [RESULT, "sample", sample.sample_name],
                         sample.sample_name, log=LOG)

    print(len(beats), [str(beat) for beat in beats])
    print(len(beat_frequencies), beat_frequencies)
    print(len(beat_pitches), beat_pitches)

    save_sheet(sheet,
               [RESULT, "sample", sample.sample_name],
               sample.sample_name, log=LOG)
