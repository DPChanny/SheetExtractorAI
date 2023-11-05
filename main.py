from DataFrameExtractor import BeatState, DataFrameExtractor
from FeatureExtractor import FeatureExtractor
from Sample import Sample
from pandas import DataFrame
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

    data_frame_extractor = DataFrameExtractor(feature_extractor.extract_stft_feature())

    beat_data_frame = data_frame_extractor.extract_beat_data_frame()
    save_data_frame(feature_extractor.sample.sample_name,
                    feature_extractor.sample.sample_name + "_beat_data_frame",
                    beat_data_frame)

    data_frame_extractor.save_beat_answer_data_frame_plot(
        DataFrame({
            "start": [0, 20, 40],
            "end": [20, 40, 60],
            "answer": [BeatState.START, BeatState.MIDDLE, BeatState.END]}),
        sample.sample_name,
        sample.sample_name + "_beat_answer_data_frame",
        sample.sample_name + "_beat_answer_data_frame")
