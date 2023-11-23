from keras import Sequential
from keras.layers import LSTM, Dense, Softmax, Bidirectional
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import argmax
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from DataFrameExtractor import START, MIDDLE, NONE
from Statics import save_plot


class BeatExtractor:
    def __init__(self, wing_length: int = 3):
        self.wing_length = wing_length
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=self.wing_length * 2 + 1,
                                          input_shape=[self.wing_length * 2 + 1, 1])))
        self.model.add(Dense(units=(self.wing_length * 2 + 1) * 2))
        self.model.add(Dense(units=self.wing_length * 2 + 1))
        self.model.add(Dense(units=3))
        self.model.add(Softmax())
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([START, MIDDLE, NONE])

    def train(self, beat_data_frame: DataFrame, beat_answer):
        beat_train = beat_data_frame.values
        beat_train = beat_train.reshape(beat_train.shape[0],
                                        beat_train.shape[1], 1)

        beat_answer_train = to_categorical(self.label_encoder.transform(beat_answer))

        history = self.model.fit(beat_train, beat_answer_train, epochs=250, validation_split=0.2)
        print(history.history)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylim(0, 1)
        save_plot("", "history_accuracy", "Accuracy")

    def predict(self, beat_data_frame: DataFrame):
        beat = beat_data_frame.values
        beat = beat.reshape(beat.shape[0],
                            beat.shape[1], 1)

        beat_predict = argmax(self.model.predict(beat), axis=1)

        beat_answer = self.label_encoder.inverse_transform(beat_predict)

        return beat_answer
