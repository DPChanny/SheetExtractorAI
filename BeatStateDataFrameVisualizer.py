from Sample import Sample
from extractor.BeatExtractor import extract_beat_states, load_beat_state_data_frame
from extractor.FeatureExtractor import extract_stft_feature, save_stft_feature_plot

LOG = True

train_samples = [Sample(["train_sample", "marimba_60_quarter_rest"],
                        "marimba_60_quarter_rest", 60, log=LOG),
                 Sample(["train_sample", "marimba_60_quarter_note"],
                        "marimba_60_quarter_note", 60, log=LOG),
                 Sample(["train_sample", "marimba_60_eighth_rest"],
                        "marimba_60_eighth_rest", 60, log=LOG),
                 Sample(["train_sample", "marimba_60_eighth_note"],
                        "marimba_60_eighth_note", 60, log=LOG),
                 Sample(["train_sample", "marimba_120_quarter_rest"],
                        "marimba_120_quarter_rest", 120, log=LOG),
                 Sample(["train_sample", "marimba_120_quarter_note"],
                        "marimba_120_quarter_note", 120, log=LOG),
                 Sample(["train_sample", "marimba_120_eighth_rest"],
                        "marimba_120_eighth_rest", 120, log=LOG),
                 Sample(["train_sample", "marimba_120_eighth_note"],
                        "marimba_120_eighth_note", 120, log=LOG),
                 Sample(["train_sample", "piano_60_quarter_rest"],
                        "piano_60_quarter_rest", 60, log=LOG),
                 Sample(["train_sample", "piano_60_quarter_note"],
                        "piano_60_quarter_note", 60, log=LOG),
                 Sample(["train_sample", "piano_60_eighth_rest"],
                        "piano_60_eighth_rest", 60, log=LOG),
                 Sample(["train_sample", "piano_60_eighth_note"],
                        "piano_60_eighth_note", 60, log=LOG),
                 Sample(["train_sample", "piano_120_quarter_rest"],
                        "piano_120_quarter_rest", 120, log=LOG),
                 Sample(["train_sample", "piano_120_quarter_note"],
                        "piano_120_quarter_note", 120, log=LOG),
                 Sample(["train_sample", "piano_120_eighth_rest"],
                        "piano_120_eighth_rest", 120, log=LOG),
                 Sample(["train_sample", "piano_120_eighth_note"],
                        "piano_120_eighth_note", 120, log=LOG)]

for train_sample in train_samples:
    train_sample_stft_feature = extract_stft_feature(train_sample, log=LOG)
    train_beat_states = extract_beat_states(train_sample,
                                            train_sample_stft_feature,
                                            load_beat_state_data_frame(["train_sample", train_sample.sample_name],
                                                                       train_sample.sample_name,
                                                                       log=LOG),
                                            log=LOG)

    save_stft_feature_plot(train_sample,
                           train_sample_stft_feature,
                           ["train_sample", train_sample.sample_name],
                           train_sample.sample_name,
                           beat_states=train_beat_states,
                           log=LOG)
