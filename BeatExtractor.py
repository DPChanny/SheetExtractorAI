from LoadSave import load_data_frame, save_plot, RESULT, SOURCE
from enum import Enum
from Sample import Sample
from pandas import DataFrame
from keras import Sequential
from numpy import argmax, array, linspace
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from FeatureExtractor import STFTFeature
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Softmax, Bidirectional


START = "start"
END = "end"
LEFT = "left"
DIFFERENCE = "difference_"
RIGHT = "right_"
STATES = "state"
VALUE = "value"


class BeatType(Enum):
    WHOLE = "whole"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"


class BeatState(Enum):
    START = "beat_start"
    MIDDLE = "beat_middle"
    NONE = "beat_none"


BeatStatusColor = {
    BeatState.START.value: "green",
    BeatState.MIDDLE.value: "blue",
    BeatState.NONE.value: "red"
}


def extract_beat_state(stft_feature: STFTFeature, beat_state_data_frame: DataFrame) -> list[str]:
    beat_state = [str(BeatState.NONE.value) for _ in range(len(stft_feature.magnitudes_sum))]

    for index in beat_state_data_frame.index:
        for i in range(beat_state_data_frame[START][index], beat_state_data_frame[END][index]):
            beat_state[i] = beat_state_data_frame[STATES][index]

    return beat_state


def load_beat_state_data_frame(directory_name: str, beat_state_data_frame_name: str):
    return load_data_frame(directory_name, beat_state_data_frame_name + ".bsdf")


def save_beat_state_plot(stft_feature: STFTFeature,
                         beat_state,
                         directory_name: str,
                         plot_name: str):

    plt.plot(linspace(start=0,
                      stop=stft_feature.duration,
                      num=len(stft_feature.magnitudes_sum)),
             stft_feature.magnitudes_sum, linewidth=0.25)

    for index in range(len(stft_feature.magnitudes_sum)):
        plt.scatter(stft_feature.duration * index / len(beat_state),
                    stft_feature.magnitudes_sum[index],
                    s=0.25, edgecolors="none", c=BeatStatusColor[beat_state[index]])

    save_plot(directory_name, plot_name + ".bst", "TIME")

    plt.plot(range(len(stft_feature.magnitudes_sum)),
             stft_feature.magnitudes_sum, linewidth=0.25)

    for index in range(len(stft_feature.magnitudes_sum)):
        plt.scatter(index,
                    stft_feature.magnitudes_sum[index],
                    s=0.25, edgecolors="none", c=BeatStatusColor[beat_state[index]])
    plt.xticks(range(0, len(stft_feature.magnitudes_sum), 5), size=1)

    save_plot(directory_name, plot_name + ".bsi", "INDEX")


def extract_beat_data_frame(stft_feature: STFTFeature, wing_length: int = 5) -> DataFrame:
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


def get_min_error_beat_type(duration: float, sample: Sample) -> tuple[BeatType, float]:
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


def extract_beat_type(beat_state: list[str], sample: Sample, stft_feature: STFTFeature) -> list[tuple[int, int, str]]:
    error_range = 1 / sample.beat_per_second / 2 / 2 ** sample.beat_per_second

    beat_type = []

    last_beat_state = BeatState.NONE.value
    last_beat_start_index = 0
    last_beat_none_index = 0

    for index, value in enumerate(beat_state):
        if value == BeatState.START.value:
            if last_beat_state == BeatState.MIDDLE.value:
                duration = (index - last_beat_start_index) / len(beat_state) * stft_feature.duration
                min_error_beat_type, min_error_beat_type_duration = get_min_error_beat_type(duration, sample)
                if abs(min_error_beat_type_duration - duration) < error_range:
                    beat_type.append((last_beat_start_index, index, min_error_beat_type.value + "_note"))
                    last_beat_state = value
                    last_beat_start_index = index
            if last_beat_state == BeatState.NONE.value:
                duration = (index - last_beat_none_index) / len(beat_state) * stft_feature.duration
                min_error_beat_type, min_error_beat_type_duration = get_min_error_beat_type(duration, sample)
                if abs(min_error_beat_type_duration - duration) < error_range:
                    beat_type.append((last_beat_none_index, index, min_error_beat_type.value + "_rest"))
                last_beat_state = value
                last_beat_start_index = index
        if value == BeatState.NONE.value:
            if last_beat_state == BeatState.MIDDLE.value:
                duration = (index - last_beat_start_index) / len(beat_state) * stft_feature.duration
                min_error_beat_type, min_error_beat_type_duration = get_min_error_beat_type(duration, sample)
                if abs(min_error_beat_type_duration - duration) < error_range:
                    beat_type.append((last_beat_start_index, index, min_error_beat_type.value + "_note"))
                    last_beat_state = value
                    last_beat_none_index = index
        if value == BeatState.MIDDLE.value:
            if last_beat_state != BeatState.NONE.value:
                last_beat_state = value

    if last_beat_state == BeatState.MIDDLE.value:
        duration = (len(beat_state) - last_beat_start_index) / len(beat_state) * stft_feature.duration
        min_error_beat_type, min_error_beat_type_duration = get_min_error_beat_type(duration, sample)
        if abs(min_error_beat_type_duration - duration) < error_range:
            beat_type.append((last_beat_start_index, len(beat_state) - 1, min_error_beat_type.value + "_note"))
    if last_beat_state == BeatState.NONE.value:
        duration = (len(beat_state) - last_beat_none_index) / len(beat_state) * stft_feature.duration
        min_error_beat_type, min_error_beat_type_duration = get_min_error_beat_type(duration, sample)
        if abs(min_error_beat_type_duration - duration) < error_range:
            beat_type.append((last_beat_none_index, len(beat_state) - 1, min_error_beat_type.value + "_rest"))

    return beat_type


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

    def save_model(self, directory_name: str, model_name: str):
        self.model.save_weights("./" + RESULT + "/" + directory_name + "/" + model_name + "/model")

    def load_model(self, directory_name: str, model_name: str):
        self.model.load_weights("./" + SOURCE + "/" + directory_name + "/" + model_name + "/model")

    def fit_model(self,
                  beat_data_frame: DataFrame,
                  beat_state: list[str],
                  epochs: int = 2 ** 10,
                  n_splits: int = 5,
                  batch_size: int = 2 ** 5):
        beat_data = beat_data_frame.values
        beat_state = array(beat_state)

        accuracy = []
        val_accuracy = []
        loss = []
        val_loss = []

        for index, (train, val) in enumerate(StratifiedKFold(n_splits=n_splits,
                                                             shuffle=True).split(beat_data,
                                                                                 beat_state)):
            print("Stratified K Fold: " + str(index + 1))

            train_beat_data, val_beat_data = beat_data[train], beat_data[val]
            train_beat_state, val_beat_state = beat_state[train], beat_state[val]

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
                                     verbose=3,
                                     callbacks=[EarlyStopping(monitor="val_loss",
                                                              patience=batch_size,
                                                              mode="min",
                                                              restore_best_weights=True,
                                                              verbose=1)])

            accuracy += history.history['accuracy']
            val_accuracy += history.history['val_accuracy']
            loss += history.history['loss']
            val_loss += history.history['val_loss']

        plt.plot(accuracy, label="accuracy")
        plt.plot(val_accuracy, label="val_accuracy")
        plt.plot(loss, label="loss")
        plt.plot(val_loss, label="val_loss")
        plt.legend()
        plt.ylim(0, 1)
        save_plot("", "train_history", "Accuracy")

    def extract_beat_state(self, beat_data_frame: DataFrame) -> list[str]:
        beat_data = beat_data_frame.values.reshape(beat_data_frame.values.shape[0],
                                                   beat_data_frame.values.shape[1], 1)

        return self.label_encoder.inverse_transform(argmax(self.model.predict(beat_data), axis=1))