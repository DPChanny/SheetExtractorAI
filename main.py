from pandas import DataFrame, concat

from Sample import Sample
from extractor.BeatExtractor import (BeatStateExtractor,
                                     save_beat_state_extractor_history_plot,
                                     extract_beat_data_frame,
                                     save_beat_data_frame,
                                     extract_beat_states,
                                     extract_beats,
                                     load_beat_state_data_frame)
from extractor.FeatureExtractor import (extract_stft_feature,
                                        save_stft_feature_plot,
                                        extract_wave_feature,
                                        save_wave_feature_plot)
from extractor.PitchExtractor import extract_beat_frequencies, extract_beat_pitches
from extractor.SheetExtractor import extract_sheet, save_sheet

LOG = True

WING_LENGTH = 5

TRAIN_BEAT_STATE_EXTRACTOR = True
BEAT_STATE_EXTRACTOR_NAME = "beat_state_extractor_0"

EPOCHS = 1024
N_SPLITS = 5
BATCH_SIZE = 32
PATIENCE = 32

SAVE_TRAIN_BEAT_DATA_FRAME = True
SAVE_BEAT_DATA_FRAME = True
SAVE_TRAIN_STFT_FEATURE_PLOT = True
SAVE_STFT_FEATURE_PLOT = True
SAVE_TRAIN_WAVE_FEATURE_PLOT = True
SAVE_WAVE_FEATURE_PLOT = True

SAVE_BEAT_STATE_EXTRACTOR_HISTORY_PLOT = True

# train_samples = [Sample(["train_sample", "marimba_60_quarter_rest"],
#                         "marimba_60_quarter_rest", 60, log=LOG),
#                  Sample(["train_sample", "marimba_60_quarter_note"],
#                         "marimba_60_quarter_note", 60, log=LOG),
#                  Sample(["train_sample", "marimba_60_eighth_rest"],
#                         "marimba_60_eighth_rest", 60, log=LOG),
#                  Sample(["train_sample", "marimba_60_eighth_note"],
#                         "marimba_60_eighth_note", 60, log=LOG),
#                  Sample(["train_sample", "marimba_120_quarter_rest"],
#                         "marimba_120_quarter_rest", 120, log=LOG),
#                  Sample(["train_sample", "marimba_120_quarter_note"],
#                         "marimba_120_quarter_note", 120, log=LOG),
#                  Sample(["train_sample", "marimba_120_eighth_rest"],
#                         "marimba_120_eighth_rest", 120, log=LOG),
#                  Sample(["train_sample", "marimba_120_eighth_note"],
#                         "marimba_120_eighth_note", 120, log=LOG),
#                  Sample(["train_sample", "piano_60_quarter_rest"],
#                         "piano_60_quarter_rest", 60, log=LOG),
#                  Sample(["train_sample", "piano_60_quarter_note"],
#                         "piano_60_quarter_note", 60, log=LOG),
#                  Sample(["train_sample", "piano_60_eighth_rest"],
#                         "piano_60_eighth_rest", 60, log=LOG),
#                  Sample(["train_sample", "piano_60_eighth_note"],
#                         "piano_60_eighth_note", 60, log=LOG),
#                  Sample(["train_sample", "piano_120_quarter_rest"],
#                         "piano_120_quarter_rest", 120, log=LOG),
#                  Sample(["train_sample", "piano_120_quarter_note"],
#                         "piano_120_quarter_note", 120, log=LOG),
#                  Sample(["train_sample", "piano_120_eighth_rest"],
#                         "piano_120_eighth_rest", 120, log=LOG),
#                  Sample(["train_sample", "piano_120_eighth_note"],
#                         "piano_120_eighth_note", 120, log=LOG)]

train_samples = [Sample(["train_sample", "marimba_60_quarter_rest"],
                        "marimba_60_quarter_rest", 60, log=LOG),
                 Sample(["train_sample", "marimba_60_quarter_note"],
                        "marimba_60_quarter_note", 60, log=LOG)]

samples = [Sample(["sample"], "marimba_60_quarter_rest", 60, log=LOG),
           Sample(["sample"], "marimba_60_quarter_note", 60, log=LOG)]

beat_state_extractor = BeatStateExtractor(WING_LENGTH)

if TRAIN_BEAT_STATE_EXTRACTOR:
    total_train_beat_data_frame = DataFrame()
    total_train_beat_states = []

    for train_sample in train_samples:
        wave_feature = extract_wave_feature(train_sample, log=LOG)
        stft_feature = extract_stft_feature(train_sample, log=LOG)
        train_beat_data_frame = extract_beat_data_frame(stft_feature, wing_length=WING_LENGTH, log=LOG)
        train_beat_states = extract_beat_states(train_sample,
                                                stft_feature,
                                                load_beat_state_data_frame(["train_sample", train_sample.sample_name],
                                                                           train_sample.sample_name, log=LOG),
                                                log=LOG)

        if SAVE_TRAIN_WAVE_FEATURE_PLOT:
            save_wave_feature_plot(train_sample,
                                   wave_feature,
                                   ["train_sample", train_sample.sample_name],
                                   train_sample.sample_name,
                                   log=LOG)

        if SAVE_TRAIN_STFT_FEATURE_PLOT:
            save_stft_feature_plot(train_sample,
                                   stft_feature,
                                   ["train_sample", train_sample.sample_name],
                                   train_sample.sample_name,
                                   beat_states=train_beat_states,
                                   log=LOG)

        if SAVE_TRAIN_BEAT_DATA_FRAME:
            save_beat_data_frame(train_beat_data_frame,
                                 ["train_sample", train_sample.sample_name],
                                 train_sample.sample_name,
                                 log=LOG)

        total_train_beat_data_frame = concat([total_train_beat_data_frame, train_beat_data_frame])
        total_train_beat_states += train_beat_states

    beat_state_extractor_history = beat_state_extractor.fit(total_train_beat_data_frame,
                                                            total_train_beat_states,
                                                            epochs=EPOCHS,
                                                            n_splits=N_SPLITS,
                                                            batch_size=BATCH_SIZE,
                                                            patience=PATIENCE,
                                                            log=LOG)

    if SAVE_BEAT_STATE_EXTRACTOR_HISTORY_PLOT:
        save_beat_state_extractor_history_plot(beat_state_extractor_history,
                                               ["beat_state_extractor", BEAT_STATE_EXTRACTOR_NAME],
                                               BEAT_STATE_EXTRACTOR_NAME, log=LOG)

    beat_state_extractor.save(["beat_state_extractor", BEAT_STATE_EXTRACTOR_NAME],
                              BEAT_STATE_EXTRACTOR_NAME, log=LOG)
else:
    beat_state_extractor.load(["beat_state_extractor", BEAT_STATE_EXTRACTOR_NAME],
                              BEAT_STATE_EXTRACTOR_NAME, log=LOG)

for sample in samples:
    wave_feature = extract_wave_feature(sample, log=LOG)
    stft_feature = extract_stft_feature(sample, log=LOG)
    beat_data_frame = extract_beat_data_frame(stft_feature, wing_length=WING_LENGTH, log=LOG)
    beat_states = beat_state_extractor.extract_beat_states(sample, beat_data_frame, log=LOG)
    beats = extract_beats(sample, beat_states, log=LOG)
    print(len(beats), [str(beat) for beat in beats])
    beat_frequencies = extract_beat_frequencies(sample, stft_feature, beats)
    print(len(beat_frequencies), beat_frequencies)
    beat_pitches = extract_beat_pitches(sample, stft_feature, beats)
    print(len(beat_pitches), beat_pitches)
    save_sheet(extract_sheet(sample, beats, beat_pitches),
               ["sample", sample.sample_name],
               sample.sample_name, log=LOG)

    if SAVE_WAVE_FEATURE_PLOT:
        save_wave_feature_plot(sample,
                               wave_feature,
                               ["sample", sample.sample_name],
                               sample.sample_name, log=LOG)

    if SAVE_STFT_FEATURE_PLOT:
        save_stft_feature_plot(sample,
                               stft_feature,
                               ["sample", sample.sample_name],
                               sample.sample_name,
                               beats=beats,
                               beat_states=beat_states,
                               log=LOG)

    if SAVE_BEAT_DATA_FRAME:
        save_beat_data_frame(beat_data_frame,
                             ["sample", sample.sample_name],
                             sample.sample_name, log=LOG)
