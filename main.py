from FeatureExtractor import Sample, FeatureExtractor

sample_names = ["piano_60", "piano_120"]

for sample_name in sample_names:
    feature_extractor = FeatureExtractor(Sample(sample_name))
    feature_extractor.extract_wave_feature(feature_extractor.sample.sampling_rate * 2)
    feature_extractor.extract_stft_feature(feature_extractor.sample.sampling_rate * 2)
