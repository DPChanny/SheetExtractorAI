from pandas import DataFrame, concat

from extraction.beat_extraction import (BeatStateExtractor,
                                        save_beat_state_extractor_history_plot,
                                        extract_beat_data_frame,
                                        save_beat_data_frame,
                                        extract_beat_states,
                                        load_beat_state_data_frame)
from extraction.feature_extraction import (extract_stft_feature,
                                           save_stft_feature_plot,
                                           extract_wave_feature,
                                           save_wave_feature_plot)
from public import Sample

LOG = True

WING_LENGTH = 5

BEAT_STATE_EXTRACTOR_NAME = "beat_state_extractor"

EPOCHS = 1024
N_SPLITS = 5
BATCH_SIZE = 32
PATIENCE = 32

SOURCE = ".\\source"
RESULT = ".\\result"

train_samples = [Sample([SOURCE, "train_sample", "marimba_60_quarter_rest"],
                        "marimba_60_quarter_rest", 60, log=LOG),
                 Sample([SOURCE, "train_sample", "marimba_60_quarter_note"],
                        "marimba_60_quarter_note", 60, log=LOG)]

beat_state_extractor = BeatStateExtractor(WING_LENGTH)
beat_state_extractor.compile()

total_train_beat_data_frame = DataFrame()
total_train_beat_states = []

for train_sample in train_samples:
    wave_feature = extract_wave_feature(train_sample, log=LOG)
    stft_feature = extract_stft_feature(train_sample, log=LOG)
    train_beat_data_frame = extract_beat_data_frame(stft_feature, wing_length=WING_LENGTH, log=LOG)
    train_beat_state_data_frame = load_beat_state_data_frame([SOURCE, "train_sample", train_sample.sample_name],
                                                             train_sample.sample_name, log=LOG)
    train_beat_states = extract_beat_states(train_sample,
                                            stft_feature,
                                            train_beat_state_data_frame,
                                            log=LOG)

    save_wave_feature_plot(train_sample,
                           wave_feature,
                           [RESULT, "train_sample", train_sample.sample_name],
                           train_sample.sample_name,
                           log=LOG)
    save_stft_feature_plot(train_sample,
                           stft_feature,
                           [RESULT, "train_sample", train_sample.sample_name],
                           train_sample.sample_name,
                           beat_states=train_beat_states,
                           log=LOG)
    save_beat_data_frame(train_beat_data_frame,
                         [RESULT, "train_sample", train_sample.sample_name],
                         train_sample.sample_name,
                         log=LOG)

    total_train_beat_data_frame = concat([total_train_beat_data_frame, train_beat_data_frame])
    total_train_beat_states += train_beat_states

beat_state_extractor_history = beat_state_extractor.fit(total_train_beat_data_frame,
                                                        total_train_beat_states,
                                                        epochs=EPOCHS,
                                                        n_splits=N_SPLITS,
                                                        batch_size=BATCH_SIZE,
                                                        patience=PATIENCE,
                                                        log=LOG)

save_beat_state_extractor_history_plot(beat_state_extractor_history,
                                       [RESULT, BEAT_STATE_EXTRACTOR_NAME],
                                       BEAT_STATE_EXTRACTOR_NAME, log=LOG)

beat_state_extractor.save([RESULT, BEAT_STATE_EXTRACTOR_NAME],
                          BEAT_STATE_EXTRACTOR_NAME, log=LOG)
