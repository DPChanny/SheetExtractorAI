from Statics import save_plot
from numpy import linspace, array
from pandas import DataFrame
from FeatureExtractor import STFTFeature
import matplotlib.pyplot as plt
from enum import Enum

START = "start"
END = "end"
LEFT = "left"
DIFFERENCE = "difference_"
RIGHT = "right_"
STATUS = "status"
VALUE = "value"


class BeatStatus(Enum):
    START = "beat_start"
    MIDDLE = "beat_middle"
    NONE = "beat_none"


BeatStatusColor = {
    BeatStatus.START.value: "green",
    BeatStatus.MIDDLE.value: "blue",
    BeatStatus.NONE.value: "red"
}


class DataFrameExtractor:
    def __init__(self, stft_feature: STFTFeature):
        self.stft_feature = stft_feature

    def get_beat_status(self, beat_status_data_frame: DataFrame) -> list[str]:
        beat_status = [str(BeatStatus.NONE.value) for _ in range(len(self.stft_feature.magnitudes_sum))]

        for index in beat_status_data_frame.index:
            for i in range(beat_status_data_frame[START][index], beat_status_data_frame[END][index] + 1):
                beat_status[i] = beat_status_data_frame[STATUS][index]

        return beat_status

    def save_beat_status_plot(self,
                              beat_status,
                              directory_name: str,
                              plot_name: str):

        plt.plot(linspace(start=0,
                          stop=self.stft_feature.duration,
                          num=len(self.stft_feature.magnitudes_sum)),
                 self.stft_feature.magnitudes_sum, linewidth=0.25)

        for i in range(len(self.stft_feature.magnitudes_sum)):
            plt.scatter(self.stft_feature.duration * i / len(beat_status),
                        self.stft_feature.magnitudes_sum[i],
                        s=0.25, c=BeatStatusColor[beat_status[i]])

        save_plot(directory_name, plot_name + "_time", "TIME")

        plt.plot(range(len(self.stft_feature.magnitudes_sum)),
                 self.stft_feature.magnitudes_sum, linewidth=0.25)

        for i in range(len(self.stft_feature.magnitudes_sum)):
            plt.scatter(i,
                        self.stft_feature.magnitudes_sum[i],
                        s=0.25, edgecolors="none", c=BeatStatusColor[beat_status[i]])
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
