from FeatureExtractor import FeatureExtractor
from Sample import Sample

samples = [("piano_60", 60), ("piano_120", 120)]

for sample in samples:
    feature_extractor = FeatureExtractor(Sample(sample[0], sample[1]))
    stft_features = feature_extractor.extract_stft_features(
        feature_extractor.sample.sampling_rate // feature_extractor.sample.bps)
