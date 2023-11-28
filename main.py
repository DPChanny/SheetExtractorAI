from pandas import DataFrame, concat
from FeatureExtractor import extract_stft_feature, save_stft_feature_plot
from BeatExtractor import BeatStateExtractor, extract_beat, save_beat_data_frame, save_beat_extractor_history_plot
from BeatExtractor import load_beat_state_data_frame, save_beat_state_plot
from BeatExtractor import extract_beat_data_frame, extract_beat_state
from Sample import Sample

LOG = True

WING_LENGTH = 5

BEAT_EXTRACTOR_NAME = "beat_extractor_0"
TRAIN_BEAT_EXTRACTOR = False

EPOCHS = 2 ** 10
N_SPLITS = 5
BATCH_SIZE = 2 ** 5

SAVE_TRAIN_BEAT_DATA_FRAME = True
SAVE_BEAT_DATA_FRAME = True

PLOT_TRAIN_STFT_FEATURE = True
PLOT_STFT_FEATURE = True
PLOT_TRAIN_BEAT_STATUS = True
PLOT_BEAT_STATUS = True
PLOT_HISTORY = True

samples = [Sample("marimba_60", 60, log=LOG),
           Sample("marimba_60_1by4", 60, log=LOG),
           Sample("marimba_60_1by8", 60, log=LOG),
           Sample("marimba_120_1by4", 120, log=LOG),
           Sample("marimba_120_1by8", 120, log=LOG),
           Sample("piano_60_1by4", 60, log=LOG),
           Sample("piano_60_1by8", 60, log=LOG),
           Sample("piano_120_1by4", 120, log=LOG),
           Sample("piano_120_1by8", 120, log=LOG)]

train_samples = [Sample("marimba_60", 60, log=LOG),
                 Sample("marimba_60_1by4", 60, log=LOG)]

beat_state_extractor = BeatStateExtractor(WING_LENGTH)

if TRAIN_BEAT_EXTRACTOR:
    total_train_beat_data_frame = DataFrame()
    total_train_beat_state = []

    for train_sample in train_samples:
        train_sample_stft_feature = extract_stft_feature(train_sample, log=LOG)
        if PLOT_TRAIN_STFT_FEATURE:
            save_stft_feature_plot(train_sample,
                                   train_sample_stft_feature,
                                   train_sample.name + "/train",
                                   train_sample.name,
                                   log=LOG)

        train_beat_data_frame = extract_beat_data_frame(train_sample_stft_feature,
                                                        wing_length=WING_LENGTH,
                                                        log=LOG)
        if SAVE_TRAIN_BEAT_DATA_FRAME:
            save_beat_data_frame(train_beat_data_frame,
                                 train_sample.name + "/train",
                                 train_sample.name,
                                 log=LOG)

        train_beat_state = extract_beat_state(train_sample,
                                              train_sample_stft_feature,
                                              load_beat_state_data_frame(train_sample.name,
                                                                         train_sample.name,
                                                                         log=LOG),
                                              log=LOG)
        if PLOT_TRAIN_BEAT_STATUS:
            save_beat_state_plot(train_sample,
                                 train_sample_stft_feature,
                                 train_beat_state,
                                 train_sample.name + "/train",
                                 train_sample.name,
                                 log=LOG)

        total_train_beat_data_frame = concat([total_train_beat_data_frame, train_beat_data_frame])
        total_train_beat_state += train_beat_state

    history = beat_state_extractor.fit(total_train_beat_data_frame,
                                       total_train_beat_state,
                                       epochs=EPOCHS,
                                       n_splits=N_SPLITS,
                                       batch_size=BATCH_SIZE,
                                       log=LOG)

    if PLOT_HISTORY:
        save_beat_extractor_history_plot(history,
                                         "beat_extractor",
                                         BEAT_EXTRACTOR_NAME,
                                         log=LOG)

    beat_state_extractor.save("beat_extractor", BEAT_EXTRACTOR_NAME, log=LOG)

if not TRAIN_BEAT_EXTRACTOR:
    beat_state_extractor.load("beat_extractor", BEAT_EXTRACTOR_NAME, log=LOG)

for sample in samples:
    sample_stft_feature = extract_stft_feature(sample, log=LOG)
    if PLOT_STFT_FEATURE:
        save_stft_feature_plot(sample,
                               sample_stft_feature,
                               sample.name,
                               sample.name,
                               log=LOG)

    beat_data_frame = extract_beat_data_frame(sample_stft_feature,
                                              wing_length=WING_LENGTH,
                                              log=LOG)
    if SAVE_BEAT_DATA_FRAME:
        save_beat_data_frame(beat_data_frame,
                             sample.name,
                             sample.name,
                             log=LOG)

    beat_state = beat_state_extractor.extract_beat_state(sample,
                                                         beat_data_frame,
                                                         log=LOG)
    if PLOT_BEAT_STATUS:
        save_beat_state_plot(sample,
                             sample_stft_feature,
                             beat_state,
                             sample.name,
                             sample.name,
                             log=LOG)

    beat = extract_beat(sample, beat_state, log=LOG)

    print(len(beat), [str(_) for _ in beat])
