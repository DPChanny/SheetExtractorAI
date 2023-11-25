from keras import Sequential
from keras.layers import LSTM, Dense, Softmax, Bidirectional
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import argmax, array
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from DataFrameExtractor import BeatStatus
from Statics import save_plot
from sklearn.model_selection import StratifiedKFold


class BeatExtractor:
    def __init__(self, wing_length: int = 3):
        self.wing_length = wing_length
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=self.wing_length * 2 + 1,
                                          input_shape=[self.wing_length * 2 + 1, 1])))
        self.model.add(Dense(units=self.wing_length * 2 + 1))
        self.model.add(Dense(units=self.wing_length * 2 + 1))
        self.model.add(Dense(units=3))
        self.model.add(Softmax())
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([BeatStatus.START.value, BeatStatus.MIDDLE.value, BeatStatus.NONE.value])

    def fit(self,
            beat_data_frame: DataFrame,
            beat_status: list[str],
            epochs: int = 50,
            n_splits: int = 5,
            batch_size: int = 5):

        beat_data = beat_data_frame.values
        beat_status = array(beat_status)

        accuracy = []
        val_accuracy = []

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
                                     epochs=epochs,
                                     batch_size=batch_size)

            accuracy += history.history['accuracy']
            val_accuracy += history.history['val_accuracy']

        plt.plot(accuracy)
        plt.plot(val_accuracy)
        plt.ylim(0, 1)
        save_plot("", "train_history", "Accuracy")

    def extract_beat_status(self, beat_data_frame: DataFrame) -> list[str]:
        beat_data = beat_data_frame.values.reshape(beat_data_frame.values.shape[0],
                                                   beat_data_frame.values.shape[1], 1)

        return self.label_encoder.inverse_transform(argmax(self.model.predict(beat_data), axis=1))

        # last_beat_answer = BeatStatus.NONE.value
        # last_beat_start = 0

        # def get_beat(start: int, end: int) -> tuple[bool, int]:
        #     duration = (end - start) / data_frame_extractor.stft_feature.duration
        #      if duration < feature_extractor.sample.beat_per_minute

        # for i in range(beat_status):
        #     if beat_status[i] == BeatStatus.START:
        #         if last_beat_answer == BeatStatus.NONE:
        #             last_beat_start = i
        #             last_beat_answer = BeatStatus.START
        #         if last_beat_answer == BeatStatus.MIDDLE:
        #             last_beat_start = i
        #             last_beat_answer = BeatStatus.START
        #     # if beat_answer[i] == BeatStatus.NONE:
        #     #     if last_beat_answer == BeatStatus.NONE:
