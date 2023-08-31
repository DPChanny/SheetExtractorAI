from FeatureExtractor import Sample, FeatureExtractor

sample_names = ["piano_60", "marimba_60", "woodwind_60", "string_60"]

for sample_name in sample_names:
    feature_extractor = FeatureExtractor(Sample(sample_name))
    feature_extractor.extract_wave_feature()
    feature_extractor.extract_stft_feature()
