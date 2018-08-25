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
from midiCanvas import MidiCanvas, ChromagramCanvas

from draggableDot import DraggableDot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class DrumAndBasslineTranscriberWindow(QtWidgets.QMainWindow):
    def __init__(self, bassline_filename, drum_filename, grid, frame_size, hop_size, fft_size, sample_rate, xlim,
                 beats=[], onsets=[], bassline_onsets=[], midi_tracks=[],drum_analysisResults=[]):

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
                                                                    bassline_onsets=bassline_onsets,
                                                                    midi_tracks=midi_tracks,
                                                                    drum_analysisResults=drum_analysisResults)

        layout = QtWidgets.QVBoxLayout(self)  # create layout out
        layout.addWidget(self.TranscriberGroupBox)  # add widget


class DrumAndBasslineTranscriberWidget(QtWidgets.QGroupBox):

    def __init__(self,  bassline_filename, drum_filename, grid, frame_size, hop_size, fft_size, sample_rate, xlim,
                 beats=[], onsets=[], bassline_onsets=[], midi_tracks=[], drum_analysisResults=[],
                 parent=None,  group_title="TRANSCRIBER"):
        self._width = 12
        self._height = 7
        self._dpi = 100

        #
        self.drum_analysisResults = drum_analysisResults
        self.grid = grid

        #   Create QGroupBox and set the parent canvas (if any)
        QtWidgets.QGroupBox.__init__(self, group_title)
        self.setParent(parent)

        self.tabs_widget = QtWidgets.QTabWidget(self)

        if bassline_filename:
            self.bassline_widget = QtWidgets.QWidget(self)
            self.tabs_widget.addTab(self.bassline_widget, "bassline")
        if drum_filename:
            self.drum_spectrogram_widget = QtWidgets.QWidget(self)
            self.drum_widget = QtWidgets.QWidget(self)
            self.tabs_widget.addTab(self.drum_spectrogram_widget, "drums spectrogram")
            self.tabs_widget.addTab(self.drum_widget, "drums transcription (onsets in bark bands)")

        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        self.resize(1200, 800)

        self.main_layout.addWidget(self.tabs_widget, 0, 0)
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setRowStretch(0, 1)

        #   ------ --------- --------- Drum Spectrogram ------ STARTS HERE
        if drum_filename:
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

            self.DrumSpectrogramCanvas = SpectrogramCanvas(parent=self.drum_spectrogram_widget, **options)

            # Draw Beats
            for beat in beats:
                self.DrumSpectrogramCanvas.get_stft_ax().axvline(x=beat, ymin=0, ymax=1000, color='g')

            # Draw Onsets
            for onset in onsets:
                self.DrumSpectrogramCanvas.get_stft_ax().scatter(onset, 50, c='red', marker='o')

            #   ------ --------- --------- Drum Spectrogram ------ ENDS HERE

            #   ------ --------- --------- Drum Transcription ------ STARTS HERE

            self.drum_onset_dots = []
            self.DrumCanvas = DrumCanvas(parent=self.drum_widget, **options)

            if self.drum_analysisResults:
               self.draw_drum_results(self.DrumCanvas.get_fig(), self.DrumCanvas.get_ax())

            for grid_line in self.grid:
                self.DrumCanvas.get_ax().axvline(x=grid_line, ymin=0, ymax=1000, color='b')

            for beat in beats:
                self.DrumCanvas.get_ax().axvline(x=beat, ymin=0, ymax=1000, color='g')

            for onset in onsets:
                self.DrumCanvas.get_ax().scatter(onset, 0.5, c='red', marker='o')

        #   ------ --------- --------- Drum Transcription ------ ENDS HERE

        #   ------ --------- --------- Bassline Transcription ------ STARTS HERE
        if bassline_filename:
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


            self.BasslineCanvas = SpectrogramCanvas(parent=self.bassline_widget, **options)

            self.InteractiveCanvas = MidiCanvas(parent=self.bassline_widget,
                                                ax=self.BasslineCanvas.get_stft_ax(),
                                                fig=self.BasslineCanvas.get_stft_fig(),
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
                self.BasslineCanvas.get_stft_ax().axvline(x=beat, ymin=0, ymax=1000, color='g')

            # Draw Onsets
            for onset in bassline_onsets:
                self.BasslineCanvas.get_stft_ax().scatter(onset, 50, c='red', marker='o')

            self.BasslineCanvas.get_stft_ax().set_title("Green: Beats, Red: Onsets, Blue: Grid")

            self.BasslineCanvas.get_stft_fig().canvas.show()
        #   ------ --------- --------- Bassline Transcription ------ ENDS HERE

        self.show()

    def draw_drum_results(self, fig, ax):
        cutoffFrequencies = self.drum_analysisResults["cutoffFrequencies"][0]
        grayscale = plt.get_cmap('binary')
        grid = self.drum_analysisResults["grid"][0]

        '''
        y_ticks = []
        y_labels = []
        
        for ix, cutoffFrequency in enumerate(cutoffFrequencies):
            y_ticks.append(ix)
            y_labels.append(str(cutoffFrequency))
            # get signal in the bark band and draw it
            signal = self.drum_analysisResults["audio_bark_fc_" + str(cutoffFrequency)][0]
            ax.plot(self.drum_analysisResults["x_time"][0], signal+ix)

            # get the onsets and relative energies detected in the bark band
            onsets = self.drum_analysisResults["onsets_bark_fc_" + str(cutoffFrequency)][0]
            energies = self.drum_analysisResults["normalized_energies_bark_fc_" + str(cutoffFrequency)][0]
            #print("energies in bark"+str(ix), energies)
            
            for energy_ix, onset in enumerate(onsets):
                self.drum_onset_dots.append(DraggableDot(fig, ax, onset, ix,
                                                         snapVerticallyFlag=True,
                                                         horizontal_snap_grid=self.grid,
                                                         defaultColor=grayscale(energies[energy_ix])))
            print("onsets in bark" + str(ix) +" loaded")
            '''


        onsets_matrix = self.drum_analysisResults["onsets_quantized_matrix"][0]
        energies_matrix = self.drum_analysisResults["energies_quantized_matrix"][0]
        #energies_matrix = energies_matrix/np.max(energies_matrix, 0)

        y_ticks = []
        y_labels = []
        for band_ix, cutoffFrequency in enumerate(cutoffFrequencies):
            y_ticks.append(band_ix)
            y_labels.append(str(cutoffFrequency))

            for grid_loc, onset in enumerate(onsets_matrix[band_ix]):
                if onset==1:
                    # print("color: ", grayscale(energies_matrix[band_ix][grid_loc]))
                    ax.scatter(grid[grid_loc], band_ix, color=grayscale(energies_matrix[band_ix][grid_loc]))

            signal = self.drum_analysisResults["audio_bark_fc_" + str(cutoffFrequency)][0]
            ax.plot(self.drum_analysisResults["x_time"][0], signal + band_ix)

        signal = self.drum_analysisResults["audio"][0]
        ax.plot(self.drum_analysisResults["x_time"][0], signal + band_ix + 1)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)

        fig.canvas.draw()

class DrumCanvas(FigureCanvas):
    # The Canvas for MIDI modification

    def __init__(self, parent=None, **options):
        #
        #   Inputs:
        #       parent                  :   parent QtCanvas
        #
        #   Options:
        #       width (int)             :   width of plot
        #       height(int)             :   height of plot
        #       dpi (int)               :   resolution of plot
        #       xlim (tuple)            :   (x0,x1)
        #       ylim (tuple)            :   (x0,x1)

        self._width = 5
        self._height = 5
        self._dpi = 100

        if "width" in options:
            self._width = options.get("width")
        if "height" in options:
            self._height = options.get("height")
        if "dpi" in options:
            self._dpi = options.get("dpi")

        # Set the x and y limits of the window
        self.xlim = (0, 20)
        self.ylim = (0, 20)


        # create figure and axes for plotting
        self.fig = Figure(figsize=(self._width, self._height), dpi=self._dpi)
        self.ax = self.fig.add_subplot(111)

        # Figure Canvas initialization
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

        # initialize variables used for calculation of spectrogram

        #initialize figure
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


        self.show()

    def get_fig(self):
        return self.fig

    def get_ax(self):
        return self.ax


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