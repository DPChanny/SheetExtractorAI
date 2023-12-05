from enum import Enum
from os import makedirs
from os.path import join
from typing import TYPE_CHECKING

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Softmax, Bidirectional
from keras.src.losses import categorical_crossentropy
from keras.src.optimizers import Adam
from keras.utils import to_categorical
from matplotlib.axes import Axes
from matplotlib.pyplot import figure
from numpy import argmax, array
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from public import FIG_HEIGHT, load_data_frame, save_data_frame, save_plot

if TYPE_CHECKING:
    from extraction.feature_extraction import STFTFeature
    from public import Sample


class BeatStateDataFrameColumn(Enum):
    START = "start"
    END = "end"
    BEAT_STATE = "beat_state"


class BeatDataFrameColumn(Enum):
    START = "start"
    END = "end"
    LEFT_DIFFERENCE = "left_difference_"
    RIGHT_DIFFERENCE = "right_difference_"
    VALUE = "value"


class BeatState(Enum):
    START = "start"
    MIDDLE = "middle"
    NONE = "none"


BeatStateColor = {
    BeatState.START: "green",
    BeatState.MIDDLE: "blue",
    BeatState.NONE: "red"
}


class BeatType(Enum):
    WHOLE = "whole"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"


class Beat:
    def __init__(self, beat_type: BeatType, start: int, end: int, note: bool):
        self.beat_type = beat_type
        self.start = start
        self.end = end
        self.note = note

    def __str__(self):
        return (self.beat_type.value +
                ("_note" if self.note else "_rest") +
                str((self.start, self.end)))


def extract_beat_states(sample: "Sample",
                        stft_feature: "STFTFeature",
                        beat_state_data_frame: DataFrame,
                        log: bool = False) -> list[BeatState]:
    if log:
        print("Extracting " + sample.sample_name + " beat state")
    beat_states = [BeatState.NONE for _ in range(len(stft_feature.magnitudes_sum))]

    for index in beat_state_data_frame.index:
        for i in range(beat_state_data_frame[BeatStateDataFrameColumn.START.value][index],
                       beat_state_data_frame[BeatStateDataFrameColumn.END.value][index]):
            beat_states[i] = BeatState(beat_state_data_frame[BeatStateDataFrameColumn.BEAT_STATE.value][index])

    return beat_states


def load_beat_state_data_frame(directory: list[str], data_frame_name: str, log: bool = False):
    return load_data_frame(directory, data_frame_name + ".bsdf", log=log)


def plot_beat_states(ax: Axes, beat_states: list[BeatState], stft_feature: "STFTFeature"):
    for index, value in enumerate(stft_feature.magnitudes_sum):
        ax.scatter(index, value, s=1.5, edgecolors="none", c=BeatStateColor[beat_states[index]])


def extract_beat_data_frame(stft_feature: "STFTFeature", wing_length: int = 5, log: bool = False) -> DataFrame:
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
        beat_data_frame[BeatDataFrameColumn.LEFT_DIFFERENCE.value + str(left)] = []
    beat_data_frame[BeatDataFrameColumn.VALUE.value] = []
    for right in range(wing_length):
        beat_data_frame[BeatDataFrameColumn.RIGHT_DIFFERENCE.value + str(right)] = []

    for i in range(len(beat_feature)):
        for right in range(wing_length):
            beat_data_frame[BeatDataFrameColumn.LEFT_DIFFERENCE.value + str(right)].append(
                get_difference(beat_feature, i + right, i + right + 1))
        beat_data_frame[BeatDataFrameColumn.VALUE.value].append(beat_feature[i])
        for left in range(wing_length):
            beat_data_frame[BeatDataFrameColumn.RIGHT_DIFFERENCE.value + str(wing_length - left - 1)].append(
                get_difference(beat_feature, i - left - 1, i - left))

    beat_data_frame = DataFrame(beat_data_frame)

    return beat_data_frame


def save_beat_data_frame(beat_data_frame: DataFrame, directory: list[str], data_frame_name: str, log: bool = False):
    save_data_frame(directory, data_frame_name + ".bdf", beat_data_frame, log=log)


def extract_beat_type(duration: float, sample: "Sample") -> tuple[BeatType, float]:
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


def extract_beats(sample: "Sample",
                  beat_states: list[BeatState],
                  log: bool = False) -> list[Beat]:
    if log:
        print("Extracting " + sample.sample_name + " beats")

    error_range = 1 / sample.beat_per_second / 2 / 5

    beats = []

    last_beat_state = BeatState.NONE
    last_start_index = 0
    last_none_index = 0

    for index, state in enumerate(beat_states):
        duration = ((index - (last_none_index if last_beat_state is BeatState.NONE else last_start_index)) /
                    len(beat_states) * sample.duration)
        beat_type, beat_type_duration = extract_beat_type(duration, sample)
        if state is BeatState.START:
            if last_beat_state is BeatState.MIDDLE:
                if abs(beat_type_duration - duration) < error_range:
                    beats.append(Beat(beat_type, last_start_index, index, True))
                    last_beat_state = state
                    last_start_index = index
            if last_beat_state is BeatState.NONE:
                if abs(beat_type_duration - duration) < error_range * 2:
                    beats.append(Beat(beat_type, last_none_index, index, False))
                last_beat_state = state
                last_start_index = index
        if state is BeatState.NONE:
            if last_beat_state is BeatState.MIDDLE:
                if abs(beat_type_duration - duration) < error_range * 2:
                    beats.append(Beat(beat_type, last_start_index, index, True))
                    last_beat_state = state
                    last_none_index = index
        if state is BeatState.MIDDLE:
            if last_beat_state is not BeatState.NONE:
                last_beat_state = state

    duration = ((len(beat_states) - last_none_index if last_beat_state is BeatState.NONE else last_start_index) /
                len(beat_states) * sample.duration)
    beat_type, beat_type_duration = extract_beat_type(duration, sample)
    if last_beat_state is BeatState.MIDDLE:
        if abs(beat_type_duration - duration) < error_range:
            beats.append(Beat(beat_type, last_start_index, len(beat_states), True))
    if last_beat_state is BeatState.NONE:
        if abs(beat_type_duration - duration) < error_range * 2:
            beats.append(Beat(beat_type, last_none_index, len(beat_states), False))

    return beats


class BeatStateExtractor:
    def __init__(self, wing_length: int = 5):
        self.wing_length = wing_length

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=self.wing_length * 2 + 1)))
        self.model.add(Dense(units=self.wing_length * 2 + 1))
        self.model.add(Dense(units=3))
        self.model.add(Softmax())
        self.model.build([self.wing_length * 2 + 1, 1, 1])

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([BeatState.START.value, BeatState.MIDDLE.value, BeatState.NONE.value])

    def compile(self):
        self.model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

    def save(self, directory: list[str], beat_state_extractor_name: str, log: bool = False):
        directory = join(*directory)
        if log:
            print("Saving " + join(*[directory, beat_state_extractor_name + ".bse.weights.h5"]))
        makedirs(directory, exist_ok=True)
        self.model.save_weights(join(*[directory, beat_state_extractor_name + ".bse.weights.h5"]))

    def load(self, directory: list[str], beat_state_extractor_name: str, log: bool = False):
        directory = join(*directory)
        if log:
            print("Loading " + join(*[directory, beat_state_extractor_name + ".bse.weights.h5"]))
        self.model.load_weights(join(*[directory, beat_state_extractor_name + ".bse.weights.h5"]))

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
        best_epochs = []
        stopped_epochs = []

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

            early_stopping = EarlyStopping(monitor="val_loss",
                                           patience=patience,
                                           mode="min",
                                           restore_best_weights=True,
                                           verbose=1 if log else 0)

            history = self.model.fit(train_beat_data,
                                     train_beat_state,
                                     validation_data=(val_beat_data,
                                                      val_beat_state),
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=2 if log else 0,
                                     callbacks=[early_stopping])

            best_epochs.append(early_stopping.best_epoch + len(accuracy))
            stopped_epochs.append(early_stopping.stopped_epoch + len(accuracy))
            accuracy += history.history['accuracy']
            val_accuracy += history.history['val_accuracy']
            loss += history.history['loss']
            val_loss += history.history['val_loss']

        history = {
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "loss": loss,
            "val_loss": val_loss,
            "best_epochs": best_epochs,
            "stopped_epochs": stopped_epochs
        }

        return history

    def extract_beat_states(self, sample: "Sample", beat_data_frame: DataFrame, log: bool = False) -> list[BeatState]:
        if log:
            print("Extracting " + sample.sample_name + " beat state")
        beat_data = beat_data_frame.values.reshape(beat_data_frame.values.shape[0],
                                                   beat_data_frame.values.shape[1], 1)

        beat_states = self.label_encoder.inverse_transform(argmax(self.model.predict(beat_data,
                                                                                     verbose=2 if log else 0),
                                                                  axis=1))

        return [BeatState(beat_state) for beat_state in beat_states]


def save_beat_state_extractor_history_plot(beat_state_extractor_history: dict,
                                           directory: list[str],
                                           plot_name: str,
                                           log: bool = False):
    fig = figure(figsize=(max(len(beat_state_extractor_history[_]) for _ in beat_state_extractor_history.keys()) / 100,
                          FIG_HEIGHT))
    fig.suptitle("Beat State Extractor History")

    history_ax = fig.add_subplot(111)

    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        history_ax.plot(beat_state_extractor_history[key], label=key, linewidth=0.5)
    for best_epoch in beat_state_extractor_history["best_epochs"]:
        history_ax.axvline(best_epoch,
                           color="green",
                           linestyle="dashdot",
                           linewidth=0.75)
    for stopped_epoch in beat_state_extractor_history["stopped_epochs"]:
        history_ax.axvline(stopped_epoch,
                           color="red",
                           linestyle="dotted",
                           linewidth=0.25)

    history_ax.set_xlabel("Epochs")

    history_ax.legend()
    history_ax.set_ylim(0, 1)
    save_plot(directory, plot_name + ".bseh", fig, log=log)


def plot_beats(ax: Axes, beats: list[Beat]):
    for beat in beats:
        ax.axvline(beat.start,
                   color="green" if beat.note else "red",
                   linestyle="dashdot",
                   linewidth=0.75)
        ax.axvline(beat.end,
                   color="green" if beat.note else "red",
                   linestyle="dotted",
                   linewidth=0.25)
