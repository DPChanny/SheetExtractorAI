from FeatureExtractor import FeatureExtractor
from Sample import Sample

sample_names = ["piano_60", "piano_120"]

for sample_name in sample_names:
    feature_extractor = FeatureExtractor(Sample(sample_name))
    feature_extractor.extract_wave_features()
    print(feature_extractor.extract_stft_features())
