from Statics import save_plot
import numpy as np
from pandas import DataFrame
from FeatureExtractor import STFTFeature
import matplotlib.pyplot as plt

S = "start"
E = "end"
LD = "left_difference_"
RD = "right_difference_"
A = "answer"


START = "beat_start"
MIDDLE = "beat_middle"
END = "beat_end"
TRANSITION = "beat_transition"
NONE = "beat_none"


BC = {
    START: "#00FF00",
    MIDDLE: "#0404B4",
    END: "#DF0101",
    TRANSITION: "#00FFFF",
    NONE: "white"
}


class DataFrameExtractor:
    def __init__(self, stft_feature: STFTFeature):
        self.stft_feature = stft_feature

    def get_beat_answer(self, beat_answer_data_frame: DataFrame) -> list[str]:
        beat_answer = [NONE for _ in range(len(self.stft_feature.magnitudes_sum))]

        for index in beat_answer_data_frame.index:
            for i in range(beat_answer_data_frame[S][index], beat_answer_data_frame[E][index] + 1):
                if beat_answer[i] != NONE:
                    beat_answer[i] = TRANSITION
                else:
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
                        s=0.5, c=BC[beat_answer[i]])

        save_plot(directory_name, plot_name + "_time", "TIME")

        plt.plot(range(len(self.stft_feature.magnitudes_sum)),
                 self.stft_feature.magnitudes_sum, linewidth=0.25)

        for i in range(len(self.stft_feature.magnitudes_sum)):
            plt.scatter(i,
                        self.stft_feature.magnitudes_sum[i],
                        s=0.5, edgecolors="none", c=BC[beat_answer[i]])
        plt.xticks(range(0, len(self.stft_feature.magnitudes_sum), 5), size=1)

        save_plot(directory_name, plot_name + "_index", "INDEX")

    def extract_beat_data_frame(self,
                                difference_count: int = 3) -> DataFrame:

        beat_data_frame = {}

        def get_difference(array, start: int, end: int):
            if start < 0 or start + 1 > len(array):
                return 0
            if end < 0 or end + 1 > len(array):
                return 0
            return array[end] - array[start]

        beat_feature = self.stft_feature.magnitudes_sum / max(self.stft_feature.magnitudes_sum)

        for ld in range(difference_count):
            beat_data_frame[LD + str(ld)] = []
        for rd in range(difference_count):
            beat_data_frame[RD + str(rd)] = []

        for i in range(len(beat_feature)):
            for rd in range(difference_count):
                beat_data_frame[RD + str(rd)].append(
                    get_difference(beat_feature, i + rd, i + rd + 1))
            for ld in range(difference_count):
                beat_data_frame[LD + str(difference_count - ld - 1)].append(
                    get_difference(beat_feature, i - ld, i - ld - 1))

        beat_data_frame = DataFrame(beat_data_frame)

        return beat_data_frame
