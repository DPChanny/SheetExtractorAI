import os
import matplotlib.pyplot as plt


def save_plot(directory_name, plot_name):
    from pylab import rcParams
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0

    print("SAVING: " + plot_name)
    os.makedirs("src/" + directory_name + "/", exist_ok=True)
    plt.savefig("src/" + directory_name + "/" + plot_name + ".png", dpi=1000)
    plt.clf()