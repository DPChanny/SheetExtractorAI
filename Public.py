from os import makedirs
from enum import Enum
from matplotlib.pyplot import close
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import arange, ndarray
from pandas import DataFrame, read_csv


class BeatState(Enum):
    START = "beat_state_start"
    MIDDLE = "beat_state_middle"
    NONE = "beat_state_none"


class BeatType(Enum):
    WHOLE = "beat_type_whole"
    HALF = "beat_type_half"
    QUARTER = "beat_type_quarter"
    EIGHTH = "beat_type_eighth"


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


def save_plot(directory_name: str, plot_name: str, plot: Figure, log: bool = False):
    if log:
        print("Saving " + plot_name + ".png")
    makedirs("./" + RESULT + "/" + directory_name + "/", exist_ok=True)
    plot.savefig("./" + RESULT + "/" + directory_name + "/" + plot_name + ".png", dpi=500)
    close(plot)


def save_data_frame(directory_name: str, data_frame_name: str, data_frame: DataFrame, log: bool = False):
    if log:
        print("Saving " + data_frame_name + ".csv")
    makedirs("./" + RESULT + "/" + directory_name + "/", exist_ok=True)
    data_frame.to_csv("./" + RESULT + "/" + directory_name + "/" + data_frame_name + ".csv")


def load_data_frame(directory_name: str, data_frame_name: str, log: bool = False) -> DataFrame:
    if log:
        print("Loading " + data_frame_name + ".csv")
    return read_csv("./" + SOURCE + "/" + directory_name + "/" + data_frame_name + ".csv")
