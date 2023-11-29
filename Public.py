import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import arange
from pandas import DataFrame, read_csv

SOURCE = "source"
RESULT = "result"

FIG_WIDTH_MULTIPLIER = 1.5
FIG_HEIGHT = 5


def set_tick(ax: Axes, index_tick, time_tick):
    ax_twin = ax.twiny()

    ax.set_xticks(arange(index_tick[0], index_tick[1], index_tick[2]))
    ax.set_xticklabels(arange(index_tick[0], index_tick[1], index_tick[2]), fontsize=5)
    ax.set_xlabel("Index")
    ax.margins(x=0, y=0.05, tight=True)
    ax.tick_params(axis='x', labelrotation=45)

    ax_twin.set_xticks(arange(time_tick[0], time_tick[1], time_tick[2]))
    ax_twin.set_xticklabels(arange(time_tick[0], time_tick[1], time_tick[2]), fontsize=5)
    ax_twin.set_xlabel("Time")
    ax_twin.margins(x=0, y=0.05, tight=True)
    ax_twin.tick_params(axis='x', labelrotation=45)


def save_plot(directory_name: str, plot_name: str, plot: Figure, log: bool = False):
    if log:
        print("Saving " + plot_name + ".png")
    os.makedirs("./" + RESULT + "/" + directory_name + "/", exist_ok=True)
    plot.savefig("./" + RESULT + "/" + directory_name + "/" + plot_name + ".png", dpi=500)
    plt.close(plot)


def save_data_frame(directory_name: str, data_frame_name: str, data_frame: DataFrame, log: bool = False):
    if log:
        print("Saving " + data_frame_name + ".csv")
    os.makedirs("./" + RESULT + "/" + directory_name + "/", exist_ok=True)
    data_frame.to_csv("./" + RESULT + "/" + directory_name + "/" + data_frame_name + ".csv")


def load_data_frame(directory_name: str, data_frame_name: str, log: bool = False) -> DataFrame:
    if log:
        print("Loading " + data_frame_name + ".csv")
    return read_csv("./" + SOURCE + "/" + directory_name + "/" + data_frame_name + ".csv")
