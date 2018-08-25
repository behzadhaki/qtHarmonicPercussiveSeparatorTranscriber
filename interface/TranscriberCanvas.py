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
import os
import json
from matplotlib.lines import Line2D

from threading import Thread

import time

import sounddevice as sd

from spectrogramCanvas import SpectrogramCanvas, InteractiveSpectrogramCanvas
from midiCanvas import MidiCanvas, ChromagramCanvas

from draggableDot import DraggableDot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import QMessageBox


class DrumAndBasslineTranscriberWindow(QtWidgets.QMainWindow):
    def __init__(self, bassline_filename, drum_filename, grid, frame_size, hop_size, fft_size, sample_rate, xlim,
                 beats=[], onsets=[], bassline_onsets=[], midi_tracks=[],drum_analysisResults=[],prefix_text=[],
                 PYIN_midi=[], YIN_times=[]):

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
                                                                    drum_analysisResults=drum_analysisResults,
                                                                    prefix_text=prefix_text,
                                                                    PYIN_midi=PYIN_midi,
                                                                    YIN_times=YIN_times)

        layout = QtWidgets.QVBoxLayout(self)  # create layout out
        layout.addWidget(self.TranscriberGroupBox)  # add widget


class DrumAndBasslineTranscriberWidget(QtWidgets.QGroupBox):
    def __init__(self,  bassline_filename, drum_filename, grid, frame_size, hop_size, fft_size, sample_rate, xlim,
                 beats=[], onsets=[], bassline_onsets=[], midi_tracks=[], drum_analysisResults=[],
                 parent=None,  group_title="TRANSCRIBER", prefix_text=[], PYIN_midi=[], YIN_times=[]):
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
            self.chroma_widget = QtWidgets.QWidget(self)
            self.tabs_widget.addTab(self.bassline_widget, "Spectrogram")
            self.tabs_widget.addTab(self.chroma_widget, "Chroma")
            self.saveFileName = os.path.join(os.path.dirname(bassline_filename), prefix_text+".txt")

        if drum_filename:
            self.drum_spectrogram_widget = QtWidgets.QWidget(self)
            self.drum_widget = QtWidgets.QWidget(self)
            self.tabs_widget.addTab(self.drum_spectrogram_widget, "drums spectrogram")
            self.tabs_widget.addTab(self.drum_widget, "drums transcription (onsets in band bands)")
            self.saveFileName = os.path.join(os.path.dirname(drum_filename), prefix_text + ".txt")

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
                       "playable": False,
                       "saveFilename": self.saveFileName}

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

            # connect save button
            self.key_pressed_cid = self.DrumCanvas.get_fig().canvas.mpl_connect('key_press_event', self.on_drum_key_press)

            if self.drum_analysisResults:
               self.draw_drum_results(self.DrumCanvas.get_fig(), self.DrumCanvas.get_ax())

            for grid_line in self.grid:
                self.DrumCanvas.get_ax().axvline(x=grid_line, ymin=0, ymax=1000, color='b')

            for beat in beats:
                self.DrumCanvas.get_ax().axvline(x=beat, ymin=0, ymax=1000, color='g')

            for onset in onsets:
                self.DrumCanvas.get_ax().scatter(onset, 0.5, c='red', marker='o')

        self.bassline_filename=None
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
                       "playable": False,
                       "saveFilename": self.saveFileName}


            self.BasslineCanvas = SpectrogramCanvas(parent=self.bassline_widget, **options)
            self.chromaCanvas = ChromaCanvas(parent=self.chroma_widget, **options)

            # Load Audio for chroma calculations
            loader = es.MonoLoader(filename=bassline_filename, sampleRate=sample_rate)
            self.audio = loader()
            xvals = np.arange(len(self.audio)) / float(sample_rate)
            xlim = [0, max(xvals)+.25]
            self.chromaCanvas.get_ax().set_xlim(xlim)

            # Calculate Chromagram
            self.chromagram = []
            hpcp = es.HPCP(size=12,  # we will need higher resolution for Key estimation
                           referenceFrequency=440,  # assume tuning frequency is 44100.
                           bandPreset=False,
                           weightType='cosine',
                           nonLinear=False,
                           windowSize=1.,
                           sampleRate=sample_rate)

            spectrum = es.Spectrum(size=fft_size)
            spectral_peaks = es.SpectralPeaks(sampleRate=sample_rate)

            for frame in es.FrameGenerator(self.audio, frameSize=8192,
                                           hopSize=hop_size, startFromZero=True):

                frame = array(frame * get_window("hann", 8192))
                freqs, mags = spectral_peaks(spectrum(frame))
                chroma = hpcp(freqs, mags)
                self.chromagram.append(chroma)

            self.chromagram = array(self.chromagram)

            self.timeAxSec = np.arange(len(self.chromagram)) * hop_size / (sample_rate)

            # plot chromagram
            pitchClasses = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
            self.chromaCanvas.get_ax().cla()
            self.chromaCanvas.get_ax().set_xlim(xlim)
            self.chromaCanvas.get_ax().set_ylim(*(-1, 13))
            y_ax = np.arange(13)
            self.chromaCanvas.get_ax().set_yticks(y_ax[:12] + .5)
            self.chromaCanvas.get_ax().set_yticklabels(pitchClasses)
            self.chromaCanvas.get_ax().pcolormesh(self.timeAxSec, y_ax, self.chromagram.T)
            self.chromaCanvas.get_ax().set_ylabel("Pitch Class")
            self.chromaCanvas.get_fig().canvas.draw()

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
                                                filename=bassline_filename,
                                                ax_chroma=self.chromaCanvas.get_ax(),
                                                fig_chroma=self.chromaCanvas.get_fig(),
                                                saveFileName=self.saveFileName)


            # Draw Beats
            for beat in beats:
                self.BasslineCanvas.get_stft_ax().axvline(x=beat, ymin=0, ymax=1000, color='g')
                self.chromaCanvas.get_ax().axvline(x=beat, ymin=0, ymax=1000, color='g')

            # Draw Onsets
            for onset in bassline_onsets:
                self.BasslineCanvas.get_stft_ax().scatter(onset, 50, c='red', marker='o')

            if PYIN_midi!=[]:
                self.BasslineCanvas.get_stft_ax().plot(YIN_times, PYIN_midi)

            self.BasslineCanvas.get_stft_ax().set_title("Green: Beats, Red: Onsets, Blue: Grid")
            self.chromaCanvas.get_ax().set_title("Green: Beats")

            # show canvases
            self.BasslineCanvas.get_stft_fig().canvas.show()
            self.chromaCanvas.get_fig().canvas.show()
        #   ------ --------- --------- Bassline Transcription ------ ENDS HERE

        self.show()

    def draw_drum_results(self, fig, ax):
        cutoffFrequencies = self.drum_analysisResults["cutoffFrequencies"][0]
        grayscale = plt.get_cmap('binary')
        grid = self.drum_analysisResults["grid"][0]
        #grid = grid
        '''
        y_ticks = []
        y_labels = []
        
        for ix, cutoffFrequency in enumerate(cutoffFrequencies):
            y_ticks.append(ix)
            y_labels.append(str(cutoffFrequency))
            # get signal in the band band and draw it
            signal = self.drum_analysisResults["audio_band_fc_" + str(cutoffFrequency)][0]
            ax.plot(self.drum_analysisResults["x_time"][0], signal+ix)

            # get the onsets and relative energies detected in the band band
            onsets = self.drum_analysisResults["onsets_band_fc_" + str(cutoffFrequency)][0]
            energies = self.drum_analysisResults["normalized_energies_band_fc_" + str(cutoffFrequency)][0]
            #print("energies in band"+str(ix), energies)
            
            for energy_ix, onset in enumerate(onsets):
                self.drum_onset_dots.append(DraggableDot(fig, ax, onset, ix,
                                                         snapVerticallyFlag=True,
                                                         horizontal_snap_grid=self.grid,
                                                         defaultColor=grayscale(energies[energy_ix])))
            print("onsets in band" + str(ix) +" loaded")
            '''


        onsets_matrix = self.drum_analysisResults["onsets_quantized_matrix"][0]
        energies_matrix = self.drum_analysisResults["energies_quantized_matrix"][0]
        #energies_matrix = energies_matrix/np.max(energies_matrix, 0)

        # save drum analysis results and onsets per step per band in a pickle

        onsets_step_band = np.transpose(onsets_matrix) # onset[1][3] --> 1 if onset in second time step and fourth band
        if len(onsets_step_band)==16:
            onsets_step_band = np.concatenate((onsets_step_band,onsets_step_band), 0)

        self.onsets_step_band = onsets_step_band

        # self.save_onset_to_txt(onsets_step_band[:32], self.saveFileName)

        # Plot onsets
        y_ticks = []
        y_labels = []
        for band_ix, cutoffFrequency in enumerate(cutoffFrequencies):
            y_ticks.append(band_ix)
            y_labels.append(str(cutoffFrequency))

            for grid_loc, onset in enumerate(onsets_matrix[band_ix]):
                if onset==1:
                    # print("color: ", grayscale(energies_matrix[band_ix][grid_loc]))
                    ax.scatter(grid[grid_loc], band_ix, color=grayscale(energies_matrix[band_ix][grid_loc]))

            signal = self.drum_analysisResults["audio_band_fc_" + str(cutoffFrequency)][0]
            ax.plot(self.drum_analysisResults["x_time"][0], signal + band_ix)

        signal = self.drum_analysisResults["audio"][0]
        ax.plot(self.drum_analysisResults["x_time"][0], signal + band_ix + 1)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)

        fig.canvas.draw()

    def on_drum_key_press(self, event):
        if event.key in ["s", "S"]:
            print("SAVING DRUM TRANSCRIPTION: ", self.saveFileName)

            self.save_onset_to_txt(self.onsets_step_band, self.saveFileName)

    def save_onset_to_txt(self, numpy_array, filename=None):
        if not (filename is None):
            if not os.path.exists(os.path.dirname(filename)):
                os.mkdir(os.path.dirname(filename))
            buttonReply = QMessageBox.question(self, "--",
                                                   "Do you want to save the transcription file?",
                                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply==QMessageBox.Yes:
                np.savetxt(filename, numpy_array, delimiter=',', fmt="%i")
        else:
            np.savetxt("drum_results.txt", numpy_array, delimiter=',', fmt="i")

        return



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
        if "filename" in options:
            self.filename = options.get("filename")
            self.audio = es.MonoLoader(filename=self.filename, sampleRate=44100)()
        else:
            self.filename = None
            self.audio = None

        self.is_playing = False

        self.play_rate = 44100 # use keys 0 to 9 to reduce speed from 100 to 90%
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

        self.key_pressed_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.show()

    def get_fig(self):
        return self.fig

    def get_ax(self):
        return self.ax

    def on_key_press(self, event):
        print (self.filename)
        if event.key=="tab":    #plays audio
            if not (self.filename is None):
                print("in")
                if not self.is_playing:
                    print("playing")
                    self.is_playing = True
                    sd.play(self.audio, self.play_rate)
                else:
                    print("stopped playing")
                    sd.stop()
                    self.is_playing = False

        if event.key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
            if event.key == "0":
                self.play_rate = 44100
            else:
                self.play_rate = 44100*float(event.key)*.1



class ChromaCanvas(FigureCanvas):
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