from pandas import read_csv
from BeatExtractor import BeatExtractor
from DataFrameExtractor import DataFrameExtractor
from FeatureExtractor import FeatureExtractor
from Sample import Sample

samples = [Sample("marimba_60_1by4", 60),
           Sample("marimba_60_1by8", 60),
           Sample("marimba_120_1by4", 120),
           Sample("marimba_120_1by8", 120),
           Sample("piano_60_1by4", 60),
           Sample("piano_60_1by8", 60),
           Sample("piano_120_1by4", 120),
           Sample("piano_120_1by8", 120),
           Sample("string_60", 60),
           Sample("string_120", 120)]

train_sample = Sample("marimba_60", 60)

train_data_frame_extractor = DataFrameExtractor(FeatureExtractor(train_sample).extract_stft_feature())

WING_LENGTH = 5

beat_extractor = BeatExtractor(wing_length=WING_LENGTH)
beat_extractor.train(train_data_frame_extractor.extract_beat_data_frame(wing_length=WING_LENGTH),
                     train_data_frame_extractor.get_beat_answer(
                             read_csv("src/" + train_sample.sample_name + "_beat_answer_data_frame.csv")))

for sample in samples:
    data_frame_extractor = DataFrameExtractor(FeatureExtractor(sample).extract_stft_feature(plot=True))
    beat_answer = beat_extractor.predict(data_frame_extractor.extract_beat_data_frame(wing_length=WING_LENGTH))
    data_frame_extractor.save_beat_answer_plot(beat_answer,
                                               sample.sample_name,
                                               sample.sample_name + "_beat_answer")
