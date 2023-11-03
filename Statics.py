import os
import matplotlib.pyplot as plt
from pylab import rcParams


def save_plot(directory_name: str, plot_name: str, plot_title: str):
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0

    print("SAVING: " + plot_name)
    os.makedirs("src/" + directory_name + "/", exist_ok=True)
    plt.title(plot_title)
    plt.savefig("src/" + directory_name + "/" + plot_name + ".png", dpi=1000)
    plt.clf()
