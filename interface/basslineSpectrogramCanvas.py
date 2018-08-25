#!/usr/bin/python
# -*-coding:Utf-8 -*
import matplotlib
matplotlib.use("Qt5Agg")
import sys
import numpy as np
from PyQt5 import QtWidgets

from spectrogramCanvas import InteractiveSpectrogramCanvas
from midiCanvas import MidiCanvas

# Module for Spectrogram plotting
import numpy as np


class InteractiveBasslineSpectrogramCanvas(InteractiveSpectrogramCanvas):
    def __init__(self, parent=None):
        if parent:
            self.parent = parent

        super(InteractiveBasslineSpectrogramCanvas, self).__init__(parent=parent)

        grid = np.arange(22)/10.0
        self.basslineMidiCanvas = MidiCanvas(parent=self.parent,
                                             ax=self.MidiCanvas.get_stft_ax(),
                                             fig=self.MidiCanvas.get_stft_fig(),
                                             horizontal_snap_grid=grid,
                                             snapVerticallyFlag=True,
                                             snap_offset_flag=True,
                                             doubleClickColor="y",
                                             xlim=self.xlim,
                                             ylim=self.stft_ylim,
                                             width=self._width,
                                             height=self.stft_height,
                                             dpi=self._dpi,
                                             x_sensitivity=.02,
                                             y_sensitivity=5,
                                             standalone=False,
                                             y_isHz=True,
                                             ax_chroma=self.ChromaCanvas.ax_chromagram,
                                             fig_chroma=self.ChromaCanvas.fig)
        #print(self.main_layout)

        #   connect callbacks for redrawing midi when the fft parameters are changed
        self.fft_size_comboBox.currentTextChanged.connect(self.redraw_midis)
        self.frame_size_comboBox.currentTextChanged.connect(self.redraw_midis)
        self.hop_size_comboBox.currentTextChanged.connect(self.redraw_midis)


    def new_midi_canvas(self, grid):
        del(self.basslineMidiCanvas)
        self.basslineMidiCanvas = MidiCanvas(parent=self.parent,
                                             ax=self.MidiCanvas.get_stft_ax(),
                                             fig=self.MidiCanvas.get_stft_fig(),
                                             horizontal_snap_grid=grid,
                                             snapVerticallyFlag=True,
                                             snap_offset_flag=True,
                                             doubleClickColor="y",
                                             xlim=self.xlim,
                                             ylim=self.stft_ylim,
                                             width=self._width,
                                             height=self.stft_height,
                                             dpi=self._dpi,
                                             x_sensitivity=.02,
                                             y_sensitivity=5,
                                             standalone=False,
                                             y_isHz=True,
                                             ax_chroma=self.ChromaCanvas.ax_chromagram,
                                             fig_chroma=self.ChromaCanvas.fig)

    def update_grid(self, grid):
        self.basslineMidiCanvas.update_grid(grid)

    def add_midi(self, onset, offset, midi):
        self.basslineMidiCanvas.add_midi_line(onset, offset, midi)

    def redraw_midis(self):
        self.basslineMidiCanvas.redraw()

    def get_shared_axis(self):
        # returns the shared x and y axis (used for synchronizing plots when zoomed or panned)
        ax = self.self.MidiCanvas.get_stft_ax()
        return ax.get_shared_x_axes(), ax.get_shared_y_axes()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    ex = InteractiveBasslineSpectrogramCanvas()
    sys.exit(app.exec_())