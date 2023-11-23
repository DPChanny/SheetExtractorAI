from Statics import save_plot
import numpy as np
from pandas import DataFrame
from FeatureExtractor import STFTFeature
import matplotlib.pyplot as plt

S = "start"
E = "end"
L = "left"
D = "difference_"
R = "right_"
A = "answer"
V = "value"


START = "beat_start"
MIDDLE = "beat_middle"
NONE = "beat_none"


BC = {
    START: "green",
    MIDDLE: "blue",
    NONE: "red"
}


class DataFrameExtractor:
    def __init__(self, stft_feature: STFTFeature):
        self.stft_feature = stft_feature

    def get_beat_answer(self, beat_answer_data_frame: DataFrame) -> list[str]:
        beat_answer = [NONE for _ in range(len(self.stft_feature.magnitudes_sum))]

        for index in beat_answer_data_frame.index:
            for i in range(beat_answer_data_frame[S][index], beat_answer_data_frame[E][index] + 1):
                beat_answer[i] = beat_answer_data_frame[A][index]

        return beat_answer

    def save_beat_answer_plot(self,
                              beat_answer,
                              directory_name: str,
                              plot_name: str):

        plt.plot(np.linspace(start=0,
                             stop=self.stft_feature.duration,
                             num=len(self.stft_feature.magnitudes_sum)),
                 self.stft_feature.magnitudes_sum, linewidth=0.25)

        for i in range(len(self.stft_feature.magnitudes_sum)):
            plt.scatter(self.stft_feature.duration * i / len(beat_answer),
                        self.stft_feature.magnitudes_sum[i],
                        s=0.25, c=BC[beat_answer[i]])

        save_plot(directory_name, plot_name + "_time", "TIME")

        plt.plot(range(len(self.stft_feature.magnitudes_sum)),
                 self.stft_feature.magnitudes_sum, linewidth=0.25)

        for i in range(len(self.stft_feature.magnitudes_sum)):
            plt.scatter(i,
                        self.stft_feature.magnitudes_sum[i],
                        s=0.25, edgecolors="none", c=BC[beat_answer[i]])
        plt.xticks(range(0, len(self.stft_feature.magnitudes_sum), 5), size=1)

        save_plot(directory_name, plot_name + "_index", "INDEX")

    def extract_beat_data_frame(self,
                                wing_length: int = 3) -> DataFrame:

        beat_data_frame = {}

        def get_difference(array, start: int, end: int):
            if start < 0 or start + 1 > len(array):
                return 0
            if end < 0 or end + 1 > len(array):
                return 0
            return array[end] - array[start]

        beat_feature = self.stft_feature.magnitudes_sum / max(self.stft_feature.magnitudes_sum)

        for left in range(wing_length):
            beat_data_frame[L + D + str(left)] = []
        beat_data_frame[V] = []
        for right in range(wing_length):
            beat_data_frame[R + D + str(right)] = []

        for i in range(len(beat_feature)):
            for right in range(wing_length):
                beat_data_frame[R + D + str(right)].append(
                    get_difference(beat_feature, i + right, i + right + 1))
            for left in range(wing_length):
                beat_data_frame[L + D + str(wing_length - left - 1)].append(
                    get_difference(beat_feature, i - left - 1, i - left))
            beat_data_frame[V].append(beat_feature[i])

        beat_data_frame = DataFrame(beat_data_frame)

        return beat_data_frame
