#!/usr/bin/python
# -*-coding:Utf-8 -*

import sys

from utils import *

import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5 import QtWidgets
from PyQt5.QtCore import *

from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

# Module for Spectrogram plotting
import numpy as np
import essentia.standard as es
from essentia import array
from scipy.fftpack import fft
from scipy.signal import get_window
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from threading import Thread

import time

import sounddevice as sd

from spectrogramCanvas import SpectrogramCanvas, InteractiveSpectrogramCanvas
from midiCanvas import MidiCanvas

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class DrumAndBasslineTranscriberWindow(QtWidgets.QMainWindow):
    def __init__(self, bassline_filename, drum_filename, grid, frame_size, hop_size, fft_size, sample_rate, xlim,
                 beats=[], onsets=[],midi_tracks=[], PYIN_midi=[], YIN_times=[]):
        #super(BasslineTranscriberWindow, self).__init__()
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")
        self.setGeometry(0, 0, 920, 780)

        self.TranscriberGroupBox = DrumAndBasslineTranscriberWidget(bassline_filename,
                                                                    drum_filename,
                                                                    grid,
                                                                    frame_size,
                                                                    hop_size,
                                                                    fft_size,
                                                                    sample_rate,
                                                                    xlim,
                                                                    beats=beats,
                                                                    onsets=onsets,
                                                                    midi_tracks=midi_tracks,
                                                                    PYIN_midi=[],
                                                                    YIN_times=[])

        layout = QtWidgets.QVBoxLayout(self)  # create layout out
        layout.addWidget(self.TranscriberGroupBox)  # add widget


class DrumAndBasslineTranscriberWidget(QtWidgets.QGroupBox):

    def __init__(self,  bassline_filename, drum_filename, grid, frame_size, hop_size, fft_size, sample_rate, xlim,
                 beats=[], onsets=[], midi_tracks=[], PYIN_midi=[], YIN_times=[],
                 parent=None,  group_title="TRANSCRIBER"):
        self._width = 12
        self._height = 7
        self._dpi = 100

        #   Create QGroupBox and set the parent canvas (if any)
        QtWidgets.QGroupBox.__init__(self, group_title)
        self.setParent(parent)

        self.tabs_widget = QtWidgets.QTabWidget(self)

        self.bassline_widget = QtWidgets.QWidget(self)
        self.drum_widget = QtWidgets.QWidget(self)

        self.tabs_widget.addTab(self.bassline_widget, "bassline")
        self.tabs_widget.addTab(self.drum_widget, "drums")

        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        self.resize(1200, 800)

        self.main_layout.addWidget(self.tabs_widget, 0, 0)
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setRowStretch(0, 1)

        #   ------ --------- --------- Drum Transcription ------ STARTS HERE
        options = {"filename": drum_filename,
                   "fft_size": fft_size,
                   "frame_size": frame_size,
                   "hop_size": hop_size,
                   "sample_rate": sample_rate,
                   "xlim": xlim,
                   "ylim": (20, 500),
                   "width": self._width,
                   "height": self._height,
                   "dpi": self._dpi,
                   "y_isHz": False,
                   "playable": False}

        self.HzCanvas = SpectrogramCanvas(parent=self.drum_widget, **options)

        #   ------ --------- --------- Drum Transcription ------ ENDS HERE

        #   ------ --------- --------- Bassline Transcription ------ STARTS HERE
        options = {"filename": bassline_filename,
                   "fft_size": fft_size,
                   "frame_size": frame_size,
                   "hop_size": hop_size,
                   "sample_rate": sample_rate,
                   "xlim": xlim,
                   "ylim": (20, 500),
                   "width": self._width,
                   "height": self._height,
                   "dpi": self._dpi,
                   "y_isHz": False,
                   "playable": False}


        self.HzCanvas = SpectrogramCanvas(parent=self.bassline_widget, **options)

        self.InteractiveCanvas = MidiCanvas(parent=self.bassline_widget,
                                            ax=self.HzCanvas.get_stft_ax(),
                                            fig=self.HzCanvas.get_stft_fig(),
                                            horizontal_snap_grid=grid,
                                            snapVerticallyFlag=True,
                                            snap_offset_flag=True,
                                            doubleClickColor="y",
                                            xlim=xlim,
                                            ylim=(20, 500),
                                            width=self._width,
                                            height=self._height,
                                            dpi=self._dpi,
                                            x_sensitivity=.02,
                                            y_sensitivity=5,
                                            standalone=False,
                                            y_isHz=False,
                                            midi_tracks=midi_tracks,
                                            filename=bassline_filename)

        # Draw Beats
        for beat in beats:
            self.HzCanvas.get_stft_ax().axvline(x=beat, ymin=0, ymax=1000, color='g')

        # Draw Onsets
        for onset in onsets:
            self.HzCanvas.get_stft_ax().scatter(onset, 50, c='red', marker='o')


        self.HzCanvas.get_stft_ax().set_title("Green: Beats, Red: Onsets, Blue: Grid")

        self.HzCanvas.get_stft_fig().canvas.show()
        #   ------ --------- --------- Bassline Transcription ------ ENDS HERE

        self.show()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    #ex = SpectrogramCanvas(filename="../audio/1/separated/kick_bass_mrfingers_median_percussive.mp3",
    #                       fft_size=2048, hop_size=128,
    #                       frame_size=512, xlim=(0, 2), ylim=(0, 1000), sample_rate=8000)
    grid = np.arange(22) / 100.0
    xlim = [0, 2]
    #ex = BasslineTranscriberWidget([], grid, 512, 126, 1024, 44100, xlim)
    ex = DrumAndBasslineTranscriberWindow([], [],grid, 512, 126, 1024, 44100, xlim)
    sys.exit(app.exec_())