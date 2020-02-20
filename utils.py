import matplotlib
import matplotlib.pyplot as plt
import simplejson as json


class LossPlot():

    def __init__(self, save_path):

        self.save_path = save_path

        matplotlib.rcParams["figure.figsize"] = (8.0, 5.0)
        matplotlib.rcParams['font.family'] = 'sans-serif'

        matplotlib.rcParams["axes.spines.left"] = True
        matplotlib.rcParams["axes.spines.top"] = True
        matplotlib.rcParams["axes.spines.right"] = True
        matplotlib.rcParams["axes.spines.bottom"] = True

        matplotlib.rcParams["axes.labelsize"] = 16

        matplotlib.rcParams["axes.titlesize"] = 14

        matplotlib.rcParams["xtick.top"] = True
        matplotlib.rcParams["xtick.direction"] = "in"
        matplotlib.rcParams["xtick.labelsize"] = 14
        matplotlib.rcParams["xtick.major.size"] = 10
        matplotlib.rcParams["xtick.minor.size"] = 5
        matplotlib.rcParams["xtick.minor.visible"] = True

        matplotlib.rcParams["ytick.right"] = True
        matplotlib.rcParams["ytick.direction"] = "in"
        matplotlib.rcParams["ytick.labelsize"] = 14
        matplotlib.rcParams["ytick.major.size"] = 10
        matplotlib.rcParams["ytick.minor.size"] = 5

        matplotlib.rcParams["lines.linewidth"] = 2

        matplotlib.rcParams["legend.fontsize"] = 14

        with open('./data/palette.json') as json_file:
            color_palette = json.load(json_file)

        self.colors = [color_palette['indigo'],
                       color_palette['red'],
                       color_palette['brown']]

        self.line_styles = [(0, ()), (0, (3, 2))]

    def draw(self, data_train, data_val, keys):
        """Plots the training and validation loss"""
        fig, ax = plt.subplots()

        plt.xlabel("Epoch", horizontalalignment='right', x=1.0)
        plt.ylabel("Loss", horizontalalignment='right', y=1.0)

        for x, (train, val, key) in enumerate(zip(data_train, data_val, keys)):
            color = self.colors[x]

            plt.plot(train,
                     linestyle=self.line_styles[0],
                     color=color['shade_800'],
                     label=key)

            plt.plot(val,
                     linestyle=self.line_styles[1],
                     color=color['shade_400'])

            plt.legend(loc="upper right",
                       frameon=False)
            plt.yscale("log")

        fig.savefig(self.save_path, bbox_inches="tight")

        plt.close(fig)
