from keras import Sequential
from keras.layers import LSTM, Dense, Softmax
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from DataFrameExtractor import START, MIDDLE, END, TRANSITION
from Statics import save_plot


class BeatExtractor:
    def __init__(self, difference_count: int = 3):
        self.rnn = Sequential()
        self.rnn.add(LSTM(units=difference_count * 2 * 2,
                          input_shape=[difference_count * 2, 1]))
        self.rnn.add(Dense(units=difference_count * 2 * 2))
        self.rnn.add(Dense(units=difference_count * 2))
        self.rnn.add(Dense(units=4))
        self.rnn.add(Softmax())
        self.rnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([START, MIDDLE, END, TRANSITION])

    def train_rnn(self, beat_data_frame: DataFrame, beat_answer):
        beat_train = beat_data_frame.values
        beat_train = beat_train.reshape(beat_train.shape[0], beat_train.shape[1], 1)

        beat_answer_train = to_categorical(self.label_encoder.transform(beat_answer))

        history = self.rnn.fit(beat_train, beat_answer_train, epochs=200, validation_split=0.2)
        print(history.history)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        save_plot("", "history_accuracy", "Accuracy")
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        save_plot("", "history_loss", "Loss")

    def predict_rnn(self, beat_data_frame: DataFrame):
        beat = beat_data_frame.values
        beat = beat.reshape(beat.shape[0], beat.shape[1], 1)

        beat_predict = argmax(self.rnn.predict(beat), axis=1)

        beat_answer = self.label_encoder.inverse_transform(beat_predict)

        return beat_answer

