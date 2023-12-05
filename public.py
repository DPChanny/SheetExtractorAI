from os import makedirs
from os.path import join
from sys import maxsize as int_max

from librosa import get_duration, load
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import close
from numpy import arange
from pandas import DataFrame, read_csv


class Sample:
    def __init__(self, directory: list[str],
                 sample_name: str,
                 beat_per_minute: int,
                 start: int = 0,
                 end: int = int_max,
                 log: bool = False):
        self.beat_per_minute = beat_per_minute
        self.beat_per_second = self.beat_per_minute / 60

        self.sample_name = sample_name

        directory = join(*directory)
        if log:
            print("Loading " + join(*[directory, sample_name + ".wav"]))
        amplitudes, self.sampling_rate = load(join(*[directory, sample_name + ".wav"]))
        self.amplitudes = amplitudes[start:min(end, len(amplitudes))]
        self.duration = get_duration(y=self.amplitudes, sr=self.sampling_rate)

        self.win_length = int(self.sampling_rate / self.beat_per_second / 16)
        self.hop_length = int(self.win_length / 4)
        self.n_fft = 4096


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
        print("Saving " + join(*[directory, plot_name + ".png"]))
    makedirs(directory, exist_ok=True)
    plot.savefig(join(*[directory, plot_name + ".png"]), dpi=500)
    close(plot)


def save_data_frame(directory: list[str], data_frame_name: str, data_frame: DataFrame, log: bool = False):
    directory = join(*directory)
    if log:
        print("Saving " + join(*[directory, data_frame_name + ".csv"]))
    makedirs(directory, exist_ok=True)
    data_frame.to_csv(join(*[directory, data_frame_name + ".csv"]))


def load_data_frame(directory: list[str], data_frame_name: str, log: bool = False) -> DataFrame:
    directory = join(*directory)
    if log:
        print("Loading " + join(*[directory, data_frame_name + ".csv"]))
    return read_csv(join(*[directory, data_frame_name + ".csv"]))
