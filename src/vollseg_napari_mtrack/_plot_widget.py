import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from qtpy import QtWidgets


class MTrackPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.canvas = FigureCanvas()
        self.canvas.figure.set_tight_layout(True)
        self.ax = self.canvas.figure.subplots(1, 1)

    def setmyPlotModel(self, model: pd.DataFrame, color="Red"):
        """Set the model. Needed so we can show/hide columns
        Args:
            model (pd.DataFrame): DataFrame to set model to.
        """
        self.myPlotModel = model

        self.rates = self.myPlotModel["Rate"]
        self.growthrates = self.rates[self.rates > 0]
        self.shrinkrates = self.rates[self.rates < 0]
        print(self.rates)
        sns.violinplot(x=self.rates, color=color, ax=self.ax)
        plt.plot(self.rates, self.rates)
        plt.show()
        self.ax.set_title("Statistics")
        self.ax.set_xlabel("Growth Rate")
        self.canvas.draw()
        self.canvas.flush_events()
