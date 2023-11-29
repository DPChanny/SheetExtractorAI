from pandas import DataFrame, concat
from FeatureExtractor import extract_stft_feature, save_stft_feature_plot, extract_wave_feature, save_wave_feature_plot
from BeatExtractor import BeatStateExtractor, extract_beats, save_beat_data_frame, save_beat_extractor_history_plot
from BeatExtractor import load_beat_state_data_frame
from BeatExtractor import extract_beat_data_frame, extract_beat_states
from Sample import Sample

LOG = True

WING_LENGTH = 5

TRAIN_BEAT_EXTRACTOR = False
BEAT_EXTRACTOR_NAME = "beat_extractor_0"

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

SAVE_BEAT_EXTRACTOR_HISTORY_PLOT = True

train_samples = [Sample(["train", "marimba_60_quarter_rest"],
                        "marimba_60_quarter_rest", 60, log=LOG),
                 Sample(["train", "marimba_60_quarter_note"],
                        "marimba_60_quarter_note", 60, log=LOG),
                 Sample(["train", "marimba_60_eighth_rest"],
                        "marimba_60_eighth_rest", 60, log=LOG),
                 Sample(["train", "marimba_60_eighth_note"],
                        "marimba_60_eighth_note", 60, log=LOG),
                 Sample(["train", "marimba_120_quarter_rest"],
                        "marimba_120_quarter_rest", 120, log=LOG),
                 Sample(["train", "marimba_120_quarter_note"],
                        "marimba_120_quarter_note", 120, log=LOG),
                 Sample(["train", "marimba_120_eighth_rest"],
                        "marimba_120_eighth_rest", 120, log=LOG),
                 Sample(["train", "marimba_120_eighth_note"],
                        "marimba_120_eighth_note", 120, log=LOG),
                 Sample(["train", "piano_60_quarter_rest"],
                        "piano_60_quarter_rest", 60, log=LOG),
                 Sample(["train", "piano_60_quarter_note"],
                        "piano_60_quarter_note", 60, log=LOG),
                 Sample(["train", "piano_60_eighth_rest"],
                        "piano_60_eighth_rest", 60, log=LOG),
                 Sample(["train", "piano_60_eighth_note"],
                        "piano_60_eighth_note", 60, log=LOG),
                 Sample(["train", "piano_120_quarter_rest"],
                        "piano_120_quarter_rest", 120, log=LOG),
                 Sample(["train", "piano_120_quarter_note"],
                        "piano_120_quarter_note", 120, log=LOG),
                 Sample(["train", "piano_120_eighth_rest"],
                        "piano_120_eighth_rest", 120, log=LOG),
                 Sample(["train", "piano_120_eighth_note"],
                        "piano_120_eighth_note", 120, log=LOG)]

samples = []

beat_state_extractor = BeatStateExtractor(WING_LENGTH)

total_train_beat_data_frame = DataFrame()
total_train_beat_states = []

for train_sample in train_samples:
    train_sample_wave_feature = extract_wave_feature(train_sample, log=LOG)
    train_sample_stft_feature = extract_stft_feature(train_sample, log=LOG)
    train_beat_data_frame = extract_beat_data_frame(train_sample_stft_feature,
                                                    wing_length=WING_LENGTH,
                                                    log=LOG)
    train_beat_states = extract_beat_states(train_sample,
                                            train_sample_stft_feature,
                                            load_beat_state_data_frame(["train", train_sample.sample_name],
                                                                       train_sample.sample_name,
                                                                       log=LOG),
                                            log=LOG)

    if SAVE_TRAIN_WAVE_FEATURE_PLOT:
        save_wave_feature_plot(train_sample,
                               train_sample_wave_feature,
                               ["train", train_sample.sample_name],
                               train_sample.sample_name,
                               log=LOG)

    if SAVE_TRAIN_STFT_FEATURE_PLOT:
        save_stft_feature_plot(train_sample,
                               train_sample_stft_feature,
                               ["train", train_sample.sample_name],
                               train_sample.sample_name,
                               beat_states=train_beat_states,
                               log=LOG)

    if SAVE_TRAIN_BEAT_DATA_FRAME:
        save_beat_data_frame(train_beat_data_frame,
                             ["train", train_sample.sample_name],
                             train_sample.sample_name,
                             log=LOG)

    total_train_beat_data_frame = concat([total_train_beat_data_frame, train_beat_data_frame])
    total_train_beat_states += train_beat_states

beat_extractor_history = beat_state_extractor.fit(total_train_beat_data_frame,
                                                  total_train_beat_states,
                                                  epochs=EPOCHS,
                                                  n_splits=N_SPLITS,
                                                  batch_size=BATCH_SIZE,
                                                  patience=PATIENCE,
                                                  log=LOG)

if SAVE_BEAT_EXTRACTOR_HISTORY_PLOT:
    save_beat_extractor_history_plot(beat_extractor_history,
                                     ["beat_extractor"],
                                     BEAT_EXTRACTOR_NAME,
                                     log=LOG)

if TRAIN_BEAT_EXTRACTOR:
    beat_state_extractor.save("beat_extractor", BEAT_EXTRACTOR_NAME, log=LOG)
else:
    beat_state_extractor.load("beat_extractor", BEAT_EXTRACTOR_NAME, log=LOG)

for sample in samples:
    sample_wave_feature = extract_wave_feature(sample, log=LOG)
    sample_stft_feature = extract_stft_feature(sample, log=LOG)
    beat_data_frame = extract_beat_data_frame(sample_stft_feature,
                                              wing_length=WING_LENGTH,
                                              log=LOG)
    beat_states = beat_state_extractor.extract_beat_states(sample,
                                                           beat_data_frame,
                                                           log=LOG)
    beat = extract_beats(sample, beat_states, log=LOG)
    print(len(beat), [str(_) for _ in beat])

    if SAVE_WAVE_FEATURE_PLOT:
        save_wave_feature_plot(sample,
                               sample_wave_feature,
                               [sample.sample_name],
                               sample.sample_name,
                               log=LOG)

    if SAVE_STFT_FEATURE_PLOT:
        save_stft_feature_plot(sample,
                               sample_stft_feature,
                               [sample.sample_name],
                               sample.sample_name,
                               beats=beat,
                               beat_states=beat_states,
                               log=LOG)

    if SAVE_BEAT_DATA_FRAME:
        save_beat_data_frame(beat_data_frame,
                             [sample.sample_name],
                             sample.sample_name,
                             log=LOG)
