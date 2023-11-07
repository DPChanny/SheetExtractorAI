from pandas import read_csv

from BeatExtractor import BeatExtractor
from DataFrameExtractor import DataFrameExtractor
from FeatureExtractor import FeatureExtractor
from Sample import Sample
from Statics import save_data_frame

samples = [Sample("marimba_60_1by4", 60),
           Sample("marimba_60_1by8", 60),
           Sample("marimba_120_1by4", 120),
           Sample("marimba_120_1by8", 120),
           Sample("piano_60_1by4", 60),
           Sample("piano_60_1by8", 60),
           Sample("piano_120_1by4", 120),
           Sample("piano_120_1by8", 120)]

# for sample in samples:
#     feature_extractor = FeatureExtractor(sample)
#
#     data_frame_extractor = DataFrameExtractor(feature_extractor.extract_stft_feature(plot=True))
#
#     beat_data_frame = data_frame_extractor.extract_beat_data_frame()
#     save_data_frame(feature_extractor.sample.sample_name,
#                     feature_extractor.sample.sample_name + "_beat_data_frame",
#                     beat_data_frame)
#
#     data_frame_extractor.save_beat_answer_data_frame_plot(
#         read_csv(sample.sample_name + "_beat_answer_data_frame.csv"),
#         sample.sample_name,
#         sample.sample_name + "_beat_answer_data_frame",
#         sample.sample_name + "_beat_answer_data_frame")

sample = samples[0]

feature_extractor = FeatureExtractor(sample)

data_frame_extractor = DataFrameExtractor(feature_extractor.extract_stft_feature())

beat_data_frame = data_frame_extractor.extract_beat_data_frame(difference_count=5)
save_data_frame(feature_extractor.sample.sample_name,
                feature_extractor.sample.sample_name + "_beat_data_frame",
                beat_data_frame)

beat_answer_data_frame = read_csv("src/" + sample.sample_name + "_beat_answer_data_frame.csv")

data_frame_extractor.save_beat_answer_plot(
    data_frame_extractor.get_beat_answer(beat_answer_data_frame),
    sample.sample_name,
    sample.sample_name + "_beat_answer_data_frame")

beat_extractor = BeatExtractor(difference_count=5)
beat_extractor.train_rnn(beat_data_frame, data_frame_extractor.get_beat_answer(beat_answer_data_frame))
beat_answer = beat_extractor.predict_rnn(beat_data_frame)
data_frame_extractor.save_beat_answer_plot(beat_answer,
                                           sample.sample_name,
                                           sample.sample_name + "_beat_answer")
