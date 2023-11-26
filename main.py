from pandas import read_csv, DataFrame, concat
from BeatStatusExtractor import BeatStatusExtractor
from DataFrameExtractor import DataFrameExtractor
from FeatureExtractor import FeatureExtractor
from Sample import Sample
from Statics import SOURCE

samples = [Sample("marimba_60", 60),
           Sample("marimba_60_1by4", 60),
           Sample("marimba_60_1by8", 60),
           Sample("marimba_120_1by4", 120),
           Sample("marimba_120_1by8", 120),
           Sample("piano_60_1by4", 60),
           Sample("piano_60_1by8", 60),
           Sample("piano_120_1by4", 120),
           Sample("piano_120_1by8", 120),
           Sample("string_60", 60),
           Sample("string_120", 120),
           Sample("woodwind_60", 60),
           Sample("woodwind_120", 120)]

train_samples = [Sample("marimba_60", 60),
                 Sample("marimba_60_1by4", 60)]

WING_LENGTH = 5
EPOCHS = 200
N_SPLITS = 5
STFT_FEATURE_PLOT = False
TRAIN_BEAT_STATUS_PLOT = False
BEAT_STATUS_PLOT = True

beat_status_extractor = BeatStatusExtractor(wing_length=WING_LENGTH)

train_beat_data_frame = DataFrame()
train_beat_status = []

for train_sample in train_samples:
    train_data_frame_extractor = DataFrameExtractor(
        FeatureExtractor(train_sample).extract_stft_feature(plot=STFT_FEATURE_PLOT))
    train_beat_data_frame = concat([train_beat_data_frame,
                                    train_data_frame_extractor.extract_beat_data_frame(wing_length=WING_LENGTH)])
    train_beat_status += train_data_frame_extractor.get_beat_status(
        read_csv("./" + SOURCE + "/" + train_sample.sample_name + "_beat_status_data_frame.csv"))

    if TRAIN_BEAT_STATUS_PLOT:
        train_data_frame_extractor.save_beat_status_plot(train_beat_status,
                                                         train_sample.sample_name,
                                                         train_sample.sample_name + "_beat_status")

beat_status_extractor.fit_model(train_beat_data_frame,
                                train_beat_status,
                                epochs=EPOCHS,
                                n_splits=N_SPLITS)

beat_status_extractor.save_model("v1", "v1.1")

# beat_status_extractor.load_model("v1", "v1.1")

for sample in samples:
    data_frame_extractor = DataFrameExtractor(FeatureExtractor(sample).extract_stft_feature(plot=STFT_FEATURE_PLOT))
    beat_data_frame = data_frame_extractor.extract_beat_data_frame(wing_length=WING_LENGTH)
    beat_status = beat_status_extractor.extract_beat_status(beat_data_frame)

    if BEAT_STATUS_PLOT:
        data_frame_extractor.save_beat_status_plot(beat_status,
                                                   sample.sample_name,
                                                   sample.sample_name + "_beat_status")
