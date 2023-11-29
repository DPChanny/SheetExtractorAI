from enum import Enum
from os import makedirs
from os.path import join

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import close
from numpy import arange, ndarray
from pandas import DataFrame, read_csv


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


class STFTFeature:
    def __init__(self,
                 magnitudes_db: ndarray,
                 magnitudes_mel_db: ndarray,
                 magnitudes_sum: ndarray):
        self.magnitudes_db = magnitudes_db
        self.magnitudes_mel_db = magnitudes_mel_db
        self.magnitudes_sum = magnitudes_sum


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


class WaveFeature:
    def __init__(self, amplitudes, amplitudes_peaks):
        self.amplitudes = amplitudes
        self.amplitudes_peaks = amplitudes_peaks


SOURCE = "source"
RESULT = "result"

FIG_WIDTH_MULTIPLIER = 1
FIG_HEIGHT = 5


def set_tick(ax: Axes, index_tick, time_tick):
    ax_twin = ax.twiny()

    ax.set_xticks(arange(index_tick[0], index_tick[1], index_tick[2]))
    ax.set_xticklabels(arange(index_tick[0], index_tick[1], index_tick[2]), fontsize=2.5)
    ax.set_xlabel("Index")
    ax.margins(x=0, y=0.05, tight=True)
    ax.tick_params(axis='x', labelrotation=45)

    ax_twin.set_xticks(arange(time_tick[0], time_tick[1], time_tick[2]))
    ax_twin.set_xticklabels(arange(time_tick[0], time_tick[1], time_tick[2]), fontsize=2.5)
    ax_twin.set_xlabel("Time")
    ax_twin.margins(x=0, y=0.05, tight=True)
    ax_twin.tick_params(axis='x', labelrotation=45)


def save_plot(directory: list[str], plot_name: str, plot: Figure, log: bool = False):
    directory = join(*directory)
    if log:
        print("Saving " + join(*[RESULT, directory, plot_name + ".png"]))
    makedirs(join(*[RESULT, directory]), exist_ok=True)
    plot.savefig(join(*[RESULT, directory, plot_name + ".png"]), dpi=500)
    close(plot)


def save_data_frame(directory: list[str], data_frame_name: str, data_frame: DataFrame, log: bool = False):
    directory = join(*directory)
    if log:
        print("Saving " + join(*[RESULT, directory, data_frame_name + ".csv"]))
    makedirs(join(*[RESULT, directory]), exist_ok=True)
    data_frame.to_csv(join(*[RESULT, directory, data_frame_name + ".csv"]))


def load_data_frame(directory: list[str], data_frame_name: str, log: bool = False) -> DataFrame:
    directory = join(*directory)
    if log:
        print("Loading " + join(*[SOURCE, directory, data_frame_name + ".csv"]))
    return read_csv(join(*[SOURCE, directory, data_frame_name + ".csv"]))
