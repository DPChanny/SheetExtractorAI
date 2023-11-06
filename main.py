from pandas import read_csv
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

for sample in samples:
    feature_extractor = FeatureExtractor(sample)

    data_frame_extractor = DataFrameExtractor(feature_extractor.extract_stft_feature(plot=True))

    beat_data_frame = data_frame_extractor.extract_beat_data_frame()
    save_data_frame(feature_extractor.sample.sample_name,
                    feature_extractor.sample.sample_name + "_beat_data_frame",
                    beat_data_frame)

    data_frame_extractor.save_beat_answer_data_frame_plot(
        read_csv(sample.sample_name + "_beat_answer_data_frame.csv"),
        sample.sample_name,
        sample.sample_name + "_beat_answer_data_frame",
        sample.sample_name + "_beat_answer_data_frame")
