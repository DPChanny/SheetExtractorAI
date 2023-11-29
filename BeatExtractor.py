from matplotlib.axes import Axes
from matplotlib.pyplot import figure
from pandas import DataFrame
from keras import Sequential
from keras.layers import LSTM, Dense, Softmax, Bidirectional
from keras.utils import to_categorical
from numpy import argmax, array
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
from FeatureExtractor import STFTFeature
from Public import load_data_frame, save_plot, BeatType, Beat, BeatState, FIG_HEIGHT
from Public import RESULT, SOURCE, save_data_frame
from Sample import Sample

START = "start"
END = "end"
LEFT = "left_"
DIFFERENCE = "difference_"
RIGHT = "right_"
BEAT_STATE = "beat_state"
VALUE = "value"


BeatStateColor = {
    BeatState.START: "green",
    BeatState.MIDDLE: "blue",
    BeatState.NONE: "red"
}


def extract_beat_states(sample: Sample,
                        stft_feature: STFTFeature,
                        beat_state_data_frame: DataFrame,
                        log: bool = False) -> list[BeatState]:
    if log:
        print("Extracting " + sample.name + " beat state")
    beat_states = [BeatState.NONE for _ in range(len(stft_feature.magnitudes_sum))]

    for index in beat_state_data_frame.index:
        for i in range(beat_state_data_frame[START][index], beat_state_data_frame[END][index]):
            beat_states[i] = BeatState(beat_state_data_frame[BEAT_STATE][index])

    return beat_states


def load_beat_state_data_frame(directory_name: str, data_frame_name: str, log: bool = False):
    return load_data_frame(directory_name, data_frame_name + ".bsdf", log=log)


def plot_beat_states(ax: Axes, beat_states: list[BeatState], stft_feature: STFTFeature):
    for index, value in enumerate(stft_feature.magnitudes_sum):
        ax.scatter(index, value, s=1.5, edgecolors="none", c=BeatStateColor[beat_states[index]])


def extract_beat_data_frame(stft_feature: STFTFeature, wing_length: int = 5, log: bool = False) -> DataFrame:
    if log:
        print("Extracting beat data frame")

    beat_data_frame = {}

    def get_difference(target_array, start, end):
        if start < 0 or start + 1 > len(target_array):
            return 0
        if end < 0 or end + 1 > len(target_array):
            return 0
        return target_array[end] - target_array[start]

    beat_feature = stft_feature.magnitudes_sum / max(stft_feature.magnitudes_sum)

    for left in range(wing_length):
        beat_data_frame[LEFT + DIFFERENCE + str(left)] = []
    beat_data_frame[VALUE] = []
    for right in range(wing_length):
        beat_data_frame[RIGHT + DIFFERENCE + str(right)] = []

    for i in range(len(beat_feature)):
        for right in range(wing_length):
            beat_data_frame[RIGHT + DIFFERENCE + str(right)].append(
                get_difference(beat_feature, i + right, i + right + 1))
        for left in range(wing_length):
            beat_data_frame[LEFT + DIFFERENCE + str(wing_length - left - 1)].append(
                get_difference(beat_feature, i - left - 1, i - left))
        beat_data_frame[VALUE].append(beat_feature[i])

    beat_data_frame = DataFrame(beat_data_frame)

    return beat_data_frame


def save_beat_data_frame(beat_data_frame: DataFrame, directory_name: str, data_frame_name: str, log: bool = False):
    save_data_frame(directory_name, data_frame_name + ".bdf", beat_data_frame, log=log)


def extract_beat_type(duration: float, sample: Sample) -> tuple[BeatType, float]:
    min_error_beat_type = BeatType.WHOLE
    min_error_beat_type_duration = 1 / sample.beat_per_second * 4
    for beat_type, beat_type_duration in [(BeatType.WHOLE, 1 / sample.beat_per_second * 4),
                                          (BeatType.HALF, 1 / sample.beat_per_second * 2),
                                          (BeatType.QUARTER, 1 / sample.beat_per_second),
                                          (BeatType.EIGHTH, 1 / sample.beat_per_second / 2)]:

        if abs(beat_type_duration - duration) < abs(min_error_beat_type_duration - duration):
            min_error_beat_type = beat_type
            min_error_beat_type_duration = beat_type_duration
    return min_error_beat_type, min_error_beat_type_duration


def extract_beats(sample: Sample,
                  beat_states: list[BeatState],
                  log: bool = False) -> list[Beat]:
    if log:
        print("Extracting " + sample.name + " beat type")

    error_range = 1 / sample.beat_per_second / 2 / 5

    beats = []

    last_beat_state = BeatState.NONE
    last_beat_start_index = 0
    last_beat_none_index = 0

    for index, state in enumerate(beat_states):
        if state is BeatState.START:
            if last_beat_state is BeatState.MIDDLE:
                duration = (index - last_beat_start_index) / len(beat_states) * sample.duration
                beat_type, beat_type_duration = extract_beat_type(duration, sample)
                if abs(beat_type_duration - duration) < error_range:
                    beats.append(Beat(beat_type, last_beat_start_index, index, True))
                    last_beat_state = state
                    last_beat_start_index = index
            if last_beat_state is BeatState.NONE:
                duration = (index - last_beat_none_index) / len(beat_states) * sample.duration
                beat_type, beat_type_duration = extract_beat_type(duration, sample)
                if abs(beat_type_duration - duration) < error_range * 2:
                    beats.append(Beat(beat_type, last_beat_none_index, index, False))
                last_beat_state = state
                last_beat_start_index = index
        if state is BeatState.NONE:
            if last_beat_state is BeatState.MIDDLE:
                duration = (index - last_beat_start_index) / len(beat_states) * sample.duration
                beat_type, beat_type_duration = extract_beat_type(duration, sample)
                if abs(beat_type_duration - duration) < error_range * 2:
                    beats.append(Beat(beat_type, last_beat_start_index, index, True))
                    last_beat_state = state
                    last_beat_none_index = index
        if state is BeatState.MIDDLE:
            if last_beat_state is not BeatState.NONE:
                last_beat_state = state

    if last_beat_state is BeatState.MIDDLE:
        duration = (len(beat_states) - last_beat_start_index) / len(beat_states) * sample.duration
        beat_type, beat_type_duration = extract_beat_type(duration, sample)
        if abs(beat_type_duration - duration) < error_range:
            beats.append(Beat(beat_type, last_beat_start_index, len(beat_states), True))
    if last_beat_state is BeatState.NONE:
        duration = (len(beat_states) - last_beat_none_index) / len(beat_states) * sample.duration
        beat_type, beat_type_duration = extract_beat_type(duration, sample)
        if abs(beat_type_duration - duration) < error_range * 2:
            beats.append(Beat(beat_type, last_beat_none_index, len(beat_states), False))

    return beats


class BeatStateExtractor:
    def __init__(self, wing_length: int = 5):
        self.wing_length = wing_length
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=self.wing_length * 2 + 1,
                                          input_shape=[self.wing_length * 2 + 1, 1])))
        self.model.add(Dense(units=self.wing_length * 2 + 1))
        self.model.add(Dense(units=3))
        self.model.add(Softmax())
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([BeatState.START.value, BeatState.MIDDLE.value, BeatState.NONE.value])

    def save(self, directory_name: str, beat_extractor_name: str, log: bool = False):
        if log:
            print("Saving " + beat_extractor_name)
        self.model.save_weights("./" + RESULT + "/" + directory_name + "/" + beat_extractor_name + "/model")

    def load(self, directory_name: str, beat_extractor_name: str, log: bool = False):
        if log:
            print("Loading " + beat_extractor_name)
        self.model.load_weights("./" + SOURCE + "/" + directory_name + "/" + beat_extractor_name + "/model")

    def fit(self,
            beat_data_frame: DataFrame,
            beat_states: list[BeatState],
            epochs: int = 1024,
            n_splits: int = 5,
            batch_size: int = 32,
            patience: int = 32,
            log: bool = False) -> dict:
        beat_data = beat_data_frame.values
        beat_states = array([beat_state.value for beat_state in beat_states])

        accuracy = []
        val_accuracy = []
        loss = []
        val_loss = []

        for index, (train, val) in enumerate(StratifiedKFold(n_splits=n_splits,
                                                             shuffle=True).split(beat_data,
                                                                                 beat_states)):
            if log:
                print("Stratified K Fold: " + str(index + 1))

            train_beat_data, val_beat_data = beat_data[train], beat_data[val]
            train_beat_state, val_beat_state = beat_states[train], beat_states[val]

            train_beat_data = train_beat_data.reshape(train_beat_data.shape[0],
                                                      train_beat_data.shape[1], 1)
            val_beat_data = val_beat_data.reshape(val_beat_data.shape[0],
                                                  val_beat_data.shape[1], 1)

            train_beat_state = to_categorical(self.label_encoder.transform(train_beat_state))
            val_beat_state = to_categorical(self.label_encoder.transform(val_beat_state))

            history = self.model.fit(train_beat_data,
                                     train_beat_state,
                                     validation_data=(val_beat_data,
                                                      val_beat_state),
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=2 if log else 0,
                                     callbacks=[EarlyStopping(monitor="val_loss",
                                                              patience=patience,
                                                              mode="min",
                                                              restore_best_weights=True,
                                                              verbose=1 if log else 0)])

            accuracy += history.history['accuracy']
            val_accuracy += history.history['val_accuracy']
            loss += history.history['loss']
            val_loss += history.history['val_loss']

        history = {
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "loss": loss,
            "val_loss": val_loss
        }

        return history

    def extract_beat_states(self, sample: Sample, beat_data_frame: DataFrame, log: bool = False) -> list[BeatState]:
        if log:
            print("Extracting " + sample.name + " beat state")
        beat_data = beat_data_frame.values.reshape(beat_data_frame.values.shape[0],
                                                   beat_data_frame.values.shape[1], 1)

        beat_states = self.label_encoder.inverse_transform(argmax(self.model.predict(beat_data,
                                                                                     verbose=2 if log else 0),
                                                                  axis=1))

        return [BeatState(beat_state) for beat_state in beat_states]


def save_beat_extractor_history_plot(beat_extractor_history: dict,
                                     directory_name: str,
                                     plot_name: str,
                                     log: bool = False):
    fig = figure(figsize=(max(len(beat_extractor_history[_]) for _ in beat_extractor_history.keys()) / 100,
                          FIG_HEIGHT))
    fig.suptitle("Beat Extractor History")

    history_ax = fig.add_subplot(111)

    for key in beat_extractor_history.keys():
        history_ax.plot(beat_extractor_history[key], label=key, linewidth=0.5)

    history_ax.legend()
    history_ax.set_ylim(0, 1)
    save_plot(directory_name, plot_name + ".beh", fig, log=log)


def plot_beats(ax: Axes, beats: list[Beat]):
    for beat in beats:
        ax.axvline(beat.start,
                   color="green" if beat.note else "red",
                   linestyle="dashdot",
                   linewidth=0.5)
        ax.axvline(beat.end,
                   color="green" if beat.note else "red",
                   linestyle="dotted",
                   linewidth=0.25)
