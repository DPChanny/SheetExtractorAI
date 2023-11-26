import sys
from enum import Enum

from keras import Sequential
from keras.layers import LSTM, Dense, Softmax, Bidirectional
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import argmax, array
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping

from DataFrameExtractor import BeatStatus
from FeatureExtractor import STFTFeature
from Sample import Sample
from Statics import save_plot, RESULT, SOURCE
from sklearn.model_selection import StratifiedKFold


class BeatType(Enum):
    WHOLE = ""
    HALF = ""
    QUARTER = ""
    EIGHTH = ""


def get_min_error_beat_type(duration: float, sample: Sample) -> tuple[BeatType, float]:
    min_error_beat_type = BeatType.WHOLE
    min_error_beat_type_duration = sys.float_info.max
    for beat_type, beat_type_duration in [(BeatType.WHOLE, 1 / sample.beat_per_second * 4),
                                          (BeatType.HALF, 1 / sample.beat_per_second * 2),
                                          (BeatType.QUARTER, 1 / sample.beat_per_second),
                                          (BeatType.EIGHTH, 1 / sample.beat_per_second / 2)]:

        if abs(beat_type_duration - duration) < min_error_beat_type_duration:
            min_error_beat_type = beat_type
            min_error_beat_type_duration = beat_type_duration - duration
    return min_error_beat_type, min_error_beat_type_duration


def extract_beat_type(beat_status: list[str], sample: Sample, stft_feature: STFTFeature):
    last_beat_answer = BeatStatus.NONE.value
    last_beat_start = 0

    for index, value in enumerate(beat_status):
        if value == BeatStatus.START:
            if last_beat_answer == BeatStatus.MIDDLE.value:
                min_error_beat_type, min_error_beat_type_duration = get_min_error_beat_type((index - last_beat_start) / stft_feature.duration, sample)

            # if last_beat_answer == BeatStatus.NONE.value:
        # if beat_answer[i] == BeatStatus.NONE:
        #     if last_beat_answer == BeatStatus.NONE:


class BeatStatusExtractor:
    def __init__(self, wing_length: int = 3):
        self.wing_length = wing_length
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=self.wing_length * 2 + 1,
                                          input_shape=[self.wing_length * 2 + 1, 1])))
        self.model.add(Dense(units=self.wing_length * 2 + 1))
        self.model.add(Dense(units=3))
        self.model.add(Softmax())
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([BeatStatus.START.value, BeatStatus.MIDDLE.value, BeatStatus.NONE.value])

    def save_model(self, directory_name: str, model_name: str):
        self.model.save_weights("./" + RESULT + "/" + directory_name + "/" + model_name + "/model")

    def load_model(self, directory_name: str, model_name: str):
        self.model.load_weights("./" + SOURCE + "/" + directory_name + "/" + model_name + "/model")

    def fit_model(self,
                  beat_data_frame: DataFrame,
                  beat_status: list[str],
                  epochs: int = 200,
                  n_splits: int = 5):
        beat_data = beat_data_frame.values
        beat_status = array(beat_status)

        accuracy = []
        val_accuracy = []
        loss = []
        val_loss = []

        for train, val in StratifiedKFold(n_splits=n_splits, shuffle=True).split(beat_data, beat_status):
            train_beat_data, val_beat_data = beat_data[train], beat_data[val]
            train_beat_status, val_beat_status = beat_status[train], beat_status[val]

            train_beat_data = train_beat_data.reshape(train_beat_data.shape[0],
                                                      train_beat_data.shape[1], 1)
            val_beat_data = val_beat_data.reshape(val_beat_data.shape[0],
                                                  val_beat_data.shape[1], 1)

            train_beat_status = to_categorical(self.label_encoder.transform(train_beat_status))
            val_beat_status = to_categorical(self.label_encoder.transform(val_beat_status))

            history = self.model.fit(train_beat_data,
                                     train_beat_status,
                                     validation_data=(val_beat_data,
                                                      val_beat_status),
                                     epochs=epochs)

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

    def extract_beat_status(self, beat_data_frame: DataFrame) -> list[str]:
        beat_data = beat_data_frame.values.reshape(beat_data_frame.values.shape[0],
                                                   beat_data_frame.values.shape[1], 1)

        return self.label_encoder.inverse_transform(argmax(self.model.predict(beat_data), axis=1))
