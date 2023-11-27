import os
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

SOURCE = "source"
RESULT = "result"


def save_plot(directory_name: str, plot_name: str, plot_title: str):
    print("SAVING: " + plot_name + ".png")
    os.makedirs("./" + RESULT + "/" + directory_name + "/", exist_ok=True)
    plt.title(plot_title)
    plt.savefig("./" + RESULT + "/" + directory_name + "/" + plot_name + ".png", dpi=1000)
    plt.clf()


def save_data_frame(directory_name: str, data_frame_name: str, data_frame: DataFrame):
    print("SAVING: " + data_frame_name + ".csv")
    os.makedirs("./" + RESULT + "/" + directory_name + "/", exist_ok=True)
    data_frame.to_csv("./" + RESULT + "/" + directory_name + "/" + data_frame_name + ".csv")


def load_data_frame(directory_name: str, data_frame_name: str) -> DataFrame:
    print("LOADING: " + data_frame_name + ".csv")
    return read_csv("./" + SOURCE + "/" + directory_name + "/" + data_frame_name + ".csv")