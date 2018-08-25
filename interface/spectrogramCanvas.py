#!/usr/bin/python
# -*-coding:Utf-8 -*

import sys

from utils import *

import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5 import QtWidgets
from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

from midiCanvas import ChromagramCanvas


class InteractiveSpectrogramCanvas(QtWidgets.QGroupBox):

    def __init__(self, parent=None,  group_title="BASSLINE SPECTROGRAM", **options):
        #   Create QGroupBox and set the parent canvas (if any)
        QtWidgets.QGroupBox.__init__(self, group_title)
        self.setParent(parent)
        self.audio_widget = QtWidgets.QWidget(self)
        self.stft_hz_widget = QtWidgets.QWidget(self)
        self.stft_midi_widget = QtWidgets.QWidget(self)
        self.stft_chromagram_widget = QtWidgets.QWidget(self)
        self.spectral_tabs_widget = QtWidgets.QTabWidget(self)

        self.spectral_tabs_widget.addTab(self.stft_hz_widget, "Hz")
        self.spectral_tabs_widget.addTab(self.stft_midi_widget, "Midi")
        self.spectral_tabs_widget.addTab(self.stft_chromagram_widget, "Chromagram")

        #   Create the grid within the group box
        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        self.resize(370, 600)

        #   Controls for plot range
        self.xlim = (0, 10)
        self.stft_ylim = (20, 500)

        #   Canvas Controls
        self._width = 4
        self.audio_height = .8
        self.stft_height = 3.5
        self._dpi = 80

        #
        self.main_layout.addWidget(self.audio_widget, 0, 0)
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setRowStretch(0, 1)
        self.main_layout.addWidget(self.spectral_tabs_widget, 1, 0, 2, 1)
        self.main_layout.setRowStretch(1, 7)

        # Create objects to be placed in each widget grid
        self.AudioCanvas = AudioWaveCanvas(parent=self.audio_widget,
                                           filename=[],
                                           xlim=self.xlim, sample_rate=44100,
                                           width=self._width, height=self.audio_height, dpi=self._dpi)

        self.HzCanvas = SpectrogramCanvas(parent=self.stft_hz_widget,
                                          filename=[],
                                          frame_size=512, xlim=self.xlim, ylim=(20, 8000), sample_rate=44100,
                                          width=self._width, height=self.stft_height, dpi=self._dpi, y_isHz=True)

        self.MidiCanvas = SpectrogramCanvas(parent=self.stft_midi_widget,
                                            filename=[],
                                            frame_size=512, xlim=self.xlim, ylim=(20, 8000), sample_rate=44100,
                                            width=self._width, height=self.stft_height, dpi=self._dpi, y_isHz=False)

        self.ChromaCanvas = ChromagramCanvas(parent=self.stft_chromagram_widget,
                                             filename=[],
                                             frame_size=512, xlim=self.xlim, sample_rate=44100,
                                             width=self._width, height=self.stft_height, dpi=self._dpi)

        # connect x and y axes to show the same region at all times
        self.join_all_axes()

        #   Create the stft control widgets
        #       FFT Size comboBox
        self.fft_size_label = QtWidgets.QLabel(self)  # fft size label
        self.fft_size_label.setText("FFT Size")
        self.fft_size_comboBox = QtWidgets.QComboBox(self)

        #       Frame Size comboBox
        self.frame_size_label = QtWidgets.QLabel(self)  # fft size label
        self.frame_size_label.setText("Frame Size")
        self.frame_size_comboBox = QtWidgets.QComboBox(self)

        #       Hop Size comboBox
        self.hop_size_label = QtWidgets.QLabel(self)  # fft size label
        self.hop_size_label.setText("Hop (x Frame)")
        self.hop_size_comboBox = QtWidgets.QComboBox(self)

        # add items and callbacks to combo boxes
        self.set_combo_box_options()
        self.connect_combobox_callbacks()

        # initialize combo boxes:
        self.fft_size_comboBox.setCurrentText("4096")
        self.frame_size_comboBox.setCurrentText("2048")
        self.hop_size_comboBox.setCurrentText("1/4")

        #   Create the grid within the group box
        self.tabs_layout = QtWidgets.QGridLayout(self.spectral_tabs_widget)
        self.tabs_layout.addWidget(self.spectral_tabs_widget, 0, 0, Qt.AlignCenter)
        self.tabs_layout.setColumnStretch(0, 10)
        self.tabs_layout.setColumnStretch(1, 10)
        self.tabs_layout.setRowStretch(0, 10)

        self.tabs_layout.addWidget(self.fft_size_label, 1, 0, Qt.AlignLeft)
        self.tabs_layout.addWidget(self.fft_size_comboBox, 1, 0, Qt.AlignRight)
        self.tabs_layout.addWidget(self.frame_size_label, 1, 1, Qt.AlignLeft)
        self.tabs_layout.addWidget(self.frame_size_comboBox, 1, 1, Qt.AlignRight)
        self.tabs_layout.addWidget(self.hop_size_label, 2, 0, Qt.AlignLeft)
        self.tabs_layout.addWidget(self.hop_size_comboBox, 2, 0, Qt.AlignRight)

        #   Show the canvas
        self.show()

    def set_combo_box_current_texts(self, fft_size, frame_size, hop_size):
        """
        sets the values of the combo boxes
        :param fft_size: must be string
        :param frame_size: must be string
        :param hop_size: must be string
        """
        self.fft_size_comboBox.setCurrentText(fft_size)
        self.frame_size_comboBox.setCurrentText(frame_size)
        self.hop_size_comboBox.setCurrentText(hop_size)

    def get_audio(self):
        return self.AudioCanvas.audio, self.AudioCanvas.sample_rate

    def get_stft(self):
        return self.HzCanvas.mX

    def join_all_axes(self):
        # connects x and y axes to show the same region at all times
        self.MidiCanvas.ax_stft.get_shared_x_axes().join(self.MidiCanvas.ax_stft, self.AudioCanvas.ax)
        self.MidiCanvas.ax_stft.get_shared_x_axes().join(self.MidiCanvas.ax_stft, self.HzCanvas.ax_stft)
        self.MidiCanvas.ax_stft.get_shared_x_axes().join(self.MidiCanvas.ax_stft, self.ChromaCanvas.ax_chromagram)
        self.MidiCanvas.ax_stft.get_shared_y_axes().join(self.MidiCanvas.ax_stft, self.HzCanvas.ax_stft)

    def set_combo_box_options(self):
        # add the items for the analysis combo boxes
        self.fft_size_comboBox.addItem("128")
        self.fft_size_comboBox.addItem("256")
        self.fft_size_comboBox.addItem("512")
        self.fft_size_comboBox.addItem("1024")
        self.fft_size_comboBox.addItem("2048")
        self.fft_size_comboBox.addItem("4096")
        self.fft_size_comboBox.addItem("8192")
        self.fft_size_comboBox.addItem("16384")

        self.frame_size_comboBox.addItem("128")
        self.frame_size_comboBox.addItem("256")
        self.frame_size_comboBox.addItem("512")
        self.frame_size_comboBox.addItem("1024")
        self.frame_size_comboBox.addItem("2048")
        self.frame_size_comboBox.addItem("4096")
        self.frame_size_comboBox.addItem("8192")
        self.frame_size_comboBox.addItem("16384")

        self.hop_size_comboBox.addItem("1/2")
        self.hop_size_comboBox.addItem("1/4")
        self.hop_size_comboBox.addItem("1/8")
        self.hop_size_comboBox.addItem("1/16")
        self.hop_size_comboBox.addItem("1")

    def connect_combobox_callbacks(self):
        # callbacks for selecting a different analysis parameter
        self.fft_size_comboBox.currentTextChanged.connect(self.fft_size_changed)    # selection callback
        self.frame_size_comboBox.currentTextChanged.connect(self.frame_size_changed)    # selection callback
        self.hop_size_comboBox.currentTextChanged.connect(self.hop_size_changed)  # selection callback

    def get_midi_ax(self):
        # returns the ax object for the midi plot
        return self.MidiCanvas.ax_stft

    def share_with_external_ax(self, ax):
        # use the same x and y axis ranges for the plots in the object with another plot out of the scope of this object
        ax.get_shared_x_axes().join(ax, self.MidiCanvas.ax_stft)
        ax.get_shared_y_axes().join(ax, self.MidiCanvas.ax_stft)

    def set_xlim(self, xlim):
        # xlim is a tuple of min and max x values
        self.MidiCanvas.update_data(xlim=xlim)
        self.HzCanvas.update_data(xlim=xlim)
        self.AudioCanvas.update_data(xlim=xlim)
        self.ChromaCanvas.update_data(xlim=xlim)

    def set_ylim(self, ylim):
        # xlim is a tuple of min and max x values
        self.MidiCanvas.update_data(ylim=ylim)
        self.HzCanvas.update_data(ylim=ylim)
        self.ChromaCanvas.update_data(ylim=ylim)

    def set_xlim_ylim(self, xlim, ylim):
        self.MidiCanvas.update_data(xlim=xlim, ylim=ylim)
        self.HzCanvas.update_data(xlim=xlim, ylim=ylim)
        self.ChromaCanvas.update_data(xlim=xlim, ylim=ylim)
        self.AudioCanvas.update_data(xlim=xlim)

    def clear_all_plots(self):
        # clear the contents of all plots
        self.AudioCanvas.ax.cla()
        self.HzCanvas.ax_stft.cla()
        self.MidiCanvas.ax_stft.cla()
        self.ChromaCanvas.ax_chromagram.cla()

        self.AudioCanvas.fig.canvas.draw()
        self.HzCanvas.fig.canvas.draw()
        self.MidiCanvas.fig.canvas.draw()
        self.ChromaCanvas.fig.canvas.draw()

        self.repaint()

    def get_audio(self):
        return self.AudioCanvas.audio, self.AudioCanvas.sample_rate

    def get_stft(self):
        return self.HzCanvas.mX

    def get_filename(self):
        return self.AudioCanvas.filename

    def set_filename(self, filename):
        if filename[-4:] == ".mp3" or filename[-4:] == ".m4a" or filename[-4:] == ".wav" or filename[-5:] == ".flac":
            self.AudioCanvas.update_data(filename=filename)
            self.HzCanvas.update_data(filename=filename)
            self.MidiCanvas.update_data(filename=filename)
            self.ChromaCanvas.update_data(filename=filename)
        else:
            self.AudioCanvas.update_data(filename=[])
            self.HzCanvas.update_data(filename=[])
            self.MidiCanvas.update_data(filename=[])
            self.ChromaCanvas.update_data(filename=[])

    def fft_size_changed(self):
        if self.MidiCanvas.filename:
            self.HzCanvas.update_data(fft_size=int(self.fft_size_comboBox.currentText()))
            self.HzCanvas.set_data_and_plots()

            self.MidiCanvas.update_data(fft_size=int(self.fft_size_comboBox.currentText()))
            self.MidiCanvas.set_data_and_plots()

            self.ChromaCanvas.update_data(fft_size=int(self.fft_size_comboBox.currentText()))
            self.ChromaCanvas.set_data_and_plots()

    def frame_size_changed(self):
        if self.MidiCanvas.filename:
            self.HzCanvas.update_data(frame_size=int(self.frame_size_comboBox.currentText()))
            self.HzCanvas.update_data(hop_size=int(float(self.frame_size_comboBox.currentText()) *
                                                      eval(self.hop_size_comboBox.currentText())))
            self.HzCanvas.set_data_and_plots()

            self.MidiCanvas.update_data(frame_size=int(self.frame_size_comboBox.currentText()))
            self.MidiCanvas.update_data(hop_size=int(float(self.frame_size_comboBox.currentText()) *
                                                   eval(self.hop_size_comboBox.currentText())))
            self.MidiCanvas.set_data_and_plots()

            self.ChromaCanvas.update_data(frame_size=int(self.frame_size_comboBox.currentText()))
            self.ChromaCanvas.update_data(hop_size=int(float(self.frame_size_comboBox.currentText()) *
                                                     eval(self.hop_size_comboBox.currentText())))
            self.ChromaCanvas.set_data_and_plots()

    def hop_size_changed(self):
        if self.MidiCanvas.filename:
            self.HzCanvas.update_data(hop_size=int(float(self.frame_size_comboBox.currentText()) *
                                                      eval(self.hop_size_comboBox.currentText())))
            self.HzCanvas.set_data_and_plots()

            self.MidiCanvas.update_data(hop_size=int(float(self.frame_size_comboBox.currentText()) *
                                                      eval(self.hop_size_comboBox.currentText())))
            self.MidiCanvas.set_data_and_plots()

            self.ChromaCanvas.update_data(hop_size=int(float(self.frame_size_comboBox.currentText()) *
                                                     eval(self.hop_size_comboBox.currentText())))
            self.ChromaCanvas.set_data_and_plots()

    def win_type_changed(self):
        if self.MidiCanvas.filename:
            self.HzCanvas.update_data(win_type=self.frame_size_comboBox.currentText())
            self.HzCanvas.set_data_and_plots()

            self.MidiCanvas.update_data(win_type=self.frame_size_comboBox.currentText())
            self.MidiCanvas.set_data_and_plots()

            self.ChromaCanvas.update_data(win_type=self.frame_size_comboBox.currentText())
            self.ChromaCanvas.set_data_and_plots()


class AudioWaveCanvas(FigureCanvas):
    def __init__(self, parent=None, **options):
        self._width = 5
        self._height = .5
        self._dpi = 100

        self.sample_rate = 44100

        # Set the x and y limits of the window
        self.xlim = (0, 20)

        self.audio = []

        self.filename = []

        # check and set provided options
        self.update_data(**options)

        # create figure and axes for plotting
        self.fig = Figure(figsize=(self._width, self._height), dpi=self._dpi)
        self.ax = self.fig.add_subplot(111)



        # Figure Canvas initialization
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if self.filename:  # if a file name is provided in constructor, plot the spectrogram
            self.load_audio()
            self.plot()

        #   variables used for plotting the moving vetrical audio scroll bar
        self.fig.canvas.mpl_connect('button_press_event', self.start_stop_play_vline)

        self.play_vline = Line2D([0, 0], [-1, 1], color="r")
        self.ax.add_line(self.play_vline)

        self._vline_thread = None  # used for audio playback vline
        self.vline_start = 0
        self.vline_current = 0

        self.vline_move_resolution = .2     # (seconds) amount to move line

        self.is_playing = False

        self.show()

    def update_data(self, **options):
        if "width" in options:
            self._width = options.get("width")
        if "height" in options:
            self._height = options.get("height")
        if "dpi" in options:
            self._dpi = options.get("dpi")
        if "xlim" in options:
            self.xlim = options.get("xlim")
        if "sample_rate" in options:
            self.sample_rate = options.get("sample_rate")
        if "filename" in options:
            self.filename = options.get("filename")
        if options:
            if self.filename:
                print("audio canv", self.filename)
                self.load_audio()
                self.plot()

    def load_audio(self):
        # loads the audio
        # apply equal-loudness filter for PredominantPitchMelodia
        loader = es.MonoLoader(filename=self.filename, sampleRate=self.sample_rate)
        self.audio = loader()
        xvals = np.arange(len(self.audio)) / float(self.sample_rate)
        self.xlim = [0, max(xvals)]

    def plot(self):
        self.ax.cla()
        # loads audio, calculates stft and updates the plot
        self.ax.add_line(self.play_vline)
        xvals = np.arange(len(self.audio))/float(self.sample_rate)
        self.xlim = [0, max(xvals)+self.sample_rate]
        self.ax.plot(xvals, self.audio)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim([-1, 1])
        self.fig.canvas.draw()
        return

    def start_stop_play_vline(self, event):
        if (event.inaxes == self.ax) and event.dblclick:
            if not self.is_playing:
                self.vline_current = self.vline_start
                self.is_playing = True
                blocksize = 2048
                self._vline_thread = Thread(target=self.move_play_vlive)
                self._vline_thread.start()
                sd.play(self.audio, self.sample_rate, blocksize=2048)
            else:
                sd.stop()
                self.is_playing = False
        return

    def move_play_vlive(self):
        while self.is_playing and (self.vline_current < (len(self.audio)/float(self.sample_rate))):
            time.sleep(self.vline_move_resolution)
            self.vline_current += self.vline_move_resolution
            self.play_vline.set_xdata([self.vline_current, self.vline_current])
            self.fig.canvas.draw()

        self.is_playing = False


class SpectrogramCanvas(FigureCanvas):
    # The Canvas for MIDI modification

    def __init__(self, parent=None, **options):
        #
        #   Inputs:
        #       parent                  :   parent QtCanvas
        #
        #   Options:
        #       filename (str)          :   full address to the audio file
        #       fft_size (int)          :   fft size for each frame (default = 1024)
        #       frame_size (int)        :   frame size for each frame (default = 1024)
        #       hop_size (int)          :   hop size between consecutive frames (default = 256)
        #       win_type (str)          :   type of the window used for framing (default = "hann")
        #       sample_rate (float)     :   sample rate to load the audio file (default = 44100)
        #       width (int)             :   width of plot
        #       height(int)             :   height of plot
        #       dpi (int)               :   resolution of plot
        #       xlim (tuple)            :   (x0,x1)
        #       ylim (tuple)            :   (x0,x1)
        #       threshold               :   (dB) threshold to remove low energy bins
        #       playable (dflt=False)   :   if true, double click on figure to play audio (disable if midi lines on top)
        #       y_isHz                  :   True to plot as Hz, False to plot as Midi Values (default=True)

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

        self.filename = []

        # create figure and axes for plotting
        self.fig = Figure(figsize=(self._width, self._height), dpi=self._dpi)
        self.ax_stft = self.fig.add_subplot(111)

        # Figure Canvas initialization
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

        # initialize variables used for calculation of spectrogram
        self.audio = []
        self.mX = array([])  # Magnitude spectrogram
        self.freqAxHz = []
        self.freqAxMidi = []
        self.timeAxSec = []
        self.fft_size = 1024
        self.frame_size = 1024
        self.hop_size = 256
        self.sample_rate = 44100
        self.win_type = "hann"
        self.window = get_window(self.win_type, self.frame_size)
        self.threshold = []
        self.playable = False
        self.y_isHz = True

        # y axis tick marks
        self.midis_ticks, self.freqs_ticks = get_midi_freq_values()

        # check and set provided options
        self.update_data(**options)





        #   variables used for plotting the moving vertical audio scroll bar
        if self.playable:
            self.fig.canvas.mpl_connect('button_press_event', self.start_stop_play_vline)

        self.play_vline = Line2D([0, 0], [0, 40000], color="r")
        self.ax_stft.add_line(self.play_vline)

        self._vline_thread = None  # used for audio playback vline
        self.vline_start = 0
        self.vline_current = 0

        self.vline_move_resolution = .1  # (seconds) amount to move line

        self.is_playing = False

        #initialize figure
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if self.filename:    # if a file name is provided in constructor, plot the spectrogram
            self.set_data_and_plots()

        self.show()

    def get_stft_fig(self):
        return self.fig

    def get_stft_ax(self):
        return self.ax_stft

    def update_data(self, **options):
        if "width" in options:
            self._width = options.get("width")
        if "height" in options:
            self._height = options.get("height")
        if "dpi" in options:
            self._dpi = options.get("dpi")
        if "xlim" in options:
            self.xlim = options.get("xlim")
        if "ylim" in options:
            self.ylim = options.get("ylim")
        if "fft_size" in options:
            self.fft_size = options.get("fft_size")
        if "frame_size" in options:
            self.frame_size = options.get("frame_size")
            self.window = get_window(self.win_type, self.frame_size)
        if "hop_size" in options:
            self.hop_size = options.get("hop_size")
        if "win_type" in options:
            self.win_type = options.get("win_type")
            self.window = get_window(self.win_type, self.frame_size)
        if "sample_rate" in options:
            self.sample_rate = options.get("sample_rate")
        if "filename" in options:
            self.filename = options.get("filename")
        if "threshold" in options:
            self.threshold = options.get("threshold")
        if "playable" in options:
            self.playable = options.get("playable")
        if "y_isHz" in options:
            self.y_isHz = options.get("y_isHz")

        if options:
            if self.filename:
                self.set_data_and_plots()

    def load_audio(self):
        # loads the audio
        # apply equal-loudness filter for PredominantPitchMelodia
        if self.filename:
            loader = es.MonoLoader(filename=self.filename, sampleRate=self.sample_rate)
            self.audio = loader()
            xvals = np.arange(len(self.audio)) / float(self.sample_rate)
            self.xlim = [0, max(xvals)]
            print("******", self.ax_stft)
            self.ax_stft.set_xlim(self.xlim)
            self.ax_stft.set_ylim(self.ylim)
        else:
            self.audio = []

    def stft(self):

        # save the results in the stft_pool
        self.mX = []
        for frame in es.FrameGenerator(self.audio, frameSize=self.frame_size,
                                       hopSize=self.hop_size, startFromZero=True):

            frame = frame*self.window
            X = fft(frame, self.fft_size)  # computing fft
            absX = np.abs(X[:int(self.fft_size / 2)])  # taking first half of the spectrum and its magnitude
            absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # getting rid of zeros before the next step
            mX = 20 * np.log10(absX)
            if self.threshold:
                mX[mX < self.threshold] = -1000
            self.mX.append(mX)

        self.mX = array(self.mX)

        self.freqAxHz = float(self.sample_rate) * np.arange(len(self.mX[0])) / float(self.fft_size)
        self.freqAxMidi = pitch2midi(self.freqAxHz, quantizePitch=False)

        self.timeAxSec = np.arange(len(self.mX))*self.hop_size/float(self.sample_rate)

    def plot_stft(self):
        if self.y_isHz:
            self.ax_stft.cla()
            self.ax_stft.set_xlim(*self.xlim)
            self.ax_stft.set_ylim(*self.ylim)
            self.ax_stft.pcolormesh(self.timeAxSec, self.freqAxHz, self.mX.T, cmap='RdBu_r')
            self.ax_stft.set_ylabel("(Hz)")  # we already handled the x-label with ax1
            self.fig.canvas.draw()
        else:
            self.ax_stft.cla()
            self.ax_stft.set_xlim(*self.xlim)
            self.ax_stft.set_yticks(self.freqs_ticks)
            self.ax_stft.set_yticklabels(self.midis_ticks)
            self.ax_stft.set_ylim(*self.ylim)
            #self.ax_stft.pcolormesh(self.timeAxSec, self.freqAxMidi, self.mX.T)
            self.ax_stft.pcolormesh(self.timeAxSec, self.freqAxHz, self.mX.T, cmap='RdBu_r')
            self.ax_stft.set_ylabel("(Midi #)")  # we already handled the x-label with ax1
            self.fig.canvas.draw()
        return

    def set_data_and_plots(self):
        # loads audio, calculates stft and updates the plot
        if self.filename:
            self.load_audio()
            self.stft()
            self.plot_stft()
        return

    def start_stop_play_vline(self, event):
        if (event.inaxes == self.ax_stft) and event.dblclick:
            if not self.is_playing:
                self.vline_current = self.vline_start
                self.is_playing = True
                self._vline_thread = Thread(target=self.move_play_vlive)
                sd.play(self.audio, self.sample_rate)
                self._vline_thread.start()
            else:
                sd.stop()
                self.is_playing = False
        return

    def move_play_vlive(self):
        while self.is_playing and (self.vline_current < (len(self.audio)/float(self.sample_rate))):
            time.sleep(self.vline_move_resolution)
            self.vline_current += self.vline_move_resolution
            self.play_vline.set_xdata([self.vline_current, self.vline_current])
            self.fig.canvas.draw()

        self.is_playing = False

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    #ex = SpectrogramCanvas(filename="../audio/1/separated/kick_bass_mrfingers_median_percussive.mp3",
    #                       fft_size=2048, hop_size=128,
    #                       frame_size=512, xlim=(0, 2), ylim=(0, 1000), sample_rate=8000)
    ex = InteractiveSpectrogramCanvas()
    sys.exit(app.exec_())