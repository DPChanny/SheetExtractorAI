from pandas import DataFrame, concat
from FeatureExtractor import extract_stft_feature, save_stft_feature_plot
from BeatExtractor import BeatStateExtractor, extract_beat_type, save_beat_data_frame, save_beat_extractor_history_plot
from BeatExtractor import load_beat_state_data_frame, save_beat_state_plot
from BeatExtractor import extract_beat_data_frame, extract_beat_state
from Sample import Sample

samples = [Sample("marimba_60", 60),
           Sample("marimba_60_1by4", 60),
           Sample("marimba_60_1by8", 60),
           Sample("marimba_120_1by4", 120),
           Sample("marimba_120_1by8", 120),
           Sample("piano_60_1by4", 60),
           Sample("piano_60_1by8", 60),
           Sample("piano_120_1by4", 120),
           Sample("piano_120_1by8", 120)]

train_samples = [Sample("marimba_60", 60),
                 Sample("marimba_60_1by4", 60)]

WING_LENGTH = 5

BEAT_EXTRACTOR_NAME = "beat_extractor_0_1"
TRAIN_BEAT_EXTRACTOR = True

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

beat_state_extractor = BeatStateExtractor(wing_length=WING_LENGTH)

if TRAIN_BEAT_EXTRACTOR:
    train_beat_data_frame = DataFrame()
    train_beat_state = []

    for train_sample in train_samples:
        train_sample_stft_feature = extract_stft_feature(train_sample)
        train_beat_data_frame = concat([train_beat_data_frame,
                                        extract_beat_data_frame(train_sample_stft_feature, wing_length=WING_LENGTH)])
        train_beat_state += extract_beat_state(train_sample_stft_feature,
                                               load_beat_state_data_frame(train_sample.name,
                                                                          train_sample.name))

        if SAVE_TRAIN_BEAT_DATA_FRAME:
            save_beat_data_frame(train_beat_data_frame,
                                 train_sample.name + "/train",
                                 train_sample.name)

        if PLOT_TRAIN_STFT_FEATURE:
            save_stft_feature_plot(train_sample,
                                   train_sample_stft_feature,
                                   train_sample.name + "/train",
                                   train_sample.name)

        if PLOT_TRAIN_BEAT_STATUS:
            save_beat_state_plot(train_sample,
                                 train_sample_stft_feature,
                                 train_beat_state,
                                 train_sample.name + "/train",
                                 train_sample.name)

    accuracy, val_accuracy, loss, val_loss = beat_state_extractor.fit(train_beat_data_frame,
                                                                      train_beat_state,
                                                                      epochs=EPOCHS,
                                                                      n_splits=N_SPLITS,
                                                                      batch_size=BATCH_SIZE)

    if PLOT_HISTORY:
        save_beat_extractor_history_plot(accuracy,
                                         val_accuracy,
                                         loss,
                                         val_loss,
                                         "beat_extractor",
                                         BEAT_EXTRACTOR_NAME)

    beat_state_extractor.save("beat_extractor", BEAT_EXTRACTOR_NAME)

if not TRAIN_BEAT_EXTRACTOR:
    beat_state_extractor.load("beat_extractor", BEAT_EXTRACTOR_NAME)

for sample in samples:
    sample_stft_feature = extract_stft_feature(sample)
    beat_data_frame = extract_beat_data_frame(sample_stft_feature, wing_length=WING_LENGTH)
    beat_state = beat_state_extractor.extract_beat_state(beat_data_frame)

    if SAVE_BEAT_DATA_FRAME:
        save_beat_data_frame(beat_data_frame,
                             sample.name,
                             sample.name)

    if PLOT_STFT_FEATURE:
        save_stft_feature_plot(sample,
                               sample_stft_feature,
                               sample.name,
                               sample.name)

    if PLOT_BEAT_STATUS:
        save_beat_state_plot(sample,
                             sample_stft_feature,
                             beat_state,
                             sample.name,
                             sample.name)

    beat_type = extract_beat_type(sample, beat_state)

    print(len(beat_type), beat_type)
