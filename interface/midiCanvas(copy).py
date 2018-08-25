#!/usr/bin/python
# -*-coding:Utf-8 -*

#https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html

import sys
import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5 import QtWidgets
from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Module for DraggableMidiLines
from draggableLine import DraggableHLine
import numpy as np

import essentia.standard as es
from essentia import array
from scipy.signal import get_window

from matplotlib.lines import Line2D

from threading import Thread

import time

import sounddevice as sd

class MidiCanvas(FigureCanvas):

    # The Canvas for MIDI modification

    def __init__(self, parent=None, **options):
        #
        #   Inputs:
        #       parent:                 : parent QtApp
        #
        #   Options:
        #       ax                      :   matplotlib ax (if midi overimposed on another plot
        #       fig                     :   matplotlib fig (if midi overimposed on another plot
        #       width (int)             :   width of plot (default = 5)
        #       height(int)             :   height of plot (default = 5)
        #       dpi (int)               :   resolution of plot (default = 100)
        #       snapVerticallyFlag      :   (bool) Set to True, if height needs to be quantized to an integer value.
        #                                                                                   (default value is False)
        #       horizontal_snap_grid    :   (1-D list or array) location of the grid to snap onsets and/or offsets
        #       snap_offset_flag (bool) :   Set True to enable snapping the offset to the grid
        #       x_sensitivity (float)   :   x distance of cursor to target for event to be recognized (default = .2)
        #       y_sensitivity (float)   :   y distance of cursor to target for event to be recognized (default = .2)
        #       line_width              :   width of the line to be plotted
        #       defaultColor            :   default color of the line when an event is not to be handled (default=blue)
        #       holdColor               :   color of line while being modified (default=red)
        #       doubleClickColor        :   color of line when double clicked on (default=green)
        #       xlim (tuple)            :   (x0,x1)
        #       ylim (tuple)            :   (x0,x1)
        #       y_isHz                  :   True: if the main axis unit is Hz (default= False)
        #

        self._width = 5
        self._height = 5
        self._dpi = 100

        self.embedded_in_spectrogram = False

        if "width" in options:
            self._width = options.get("width")
        if "height" in options:
            self._height = options.get("height")
        if "dpi" in options:
            self._dpi = options.get("dpi")
        if "y_isHz" in options:
            self.y_isHz = options.get("y_isHz")

        #check if y axis is Hz
        if "y_isHz" in options:
            self.y_isHz = options.get("y_isHz")
        else:
            self.y_isHz = False

        # create figure and axes for plotting
        if parent:
            self.ax = options.get("ax")
            self.fig = options.get("fig")
            options.pop("ax")
            options.pop("fig")
        else:
            self.fig = Figure(figsize=(self._width, self._height), dpi=self._dpi)
            self.ax = self.fig.add_subplot(111)

        # parameters used for interactive addition of midi lines
        self.horizontal_snap_grid = []

        if "horizontal_snap_grid" in options:
            self.horizontal_snap_grid = np.array(options.get("horizontal_snap_grid"))

        # save options for plotting lines
        self.options = options

        # Figure Canvas initialization in case stand alone midi plot
        if not parent:
            FigureCanvas.__init__(self, self.fig)
            self.setParent(parent)

            FigureCanvas.setSizePolicy(self,
                                       QtWidgets.QSizePolicy.Expanding,
                                       QtWidgets.QSizePolicy.Expanding)
            FigureCanvas.updateGeometry(self)

        self.fig.canvas.setFocusPolicy(Qt.ClickFocus)
        self.fig.canvas.setFocus()

        # Set the x and y limits of the window
        self.xlim = (0, 20)
        self.ylim = (0, 20)

        if "xlim" in options:
            self.xlim = options.get("xlim")

        if "ylim" in options:
            self.ylim = options.get("ylim")

        if not parent:
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(*self.ylim)

        # To store the 2 draggable points
        self.midi_draggableLines = []

        # Flags for interactively adding lines #
        # To add a line, hold "A" on keyboard, and double click on onset location and offset location)
        self.onsetDoubleClickFlag = False
        self.addLineKeyPressedFlag = False
        self.holdFlag = False

        # identifiers for connected event handler
        self.key_pressed_cid = []
        self.key_released_cid = []
        self.mouse_pressed_cid = []

        # variables for new lines to be added
        self.newLineParameters = []     #   should be stored as [onset, midi value, offset]

        # draw grid
        if self.horizontal_snap_grid != []:
            self.draw_grid()
        # connect event handlers
        self.connect()

        self.fig.canvas.show()

    def change_y_isHz(self, _is_hz):
        # set true or false
        self.y_isHz = _is_hz
    def connect(self):
        # connects the event handling functions
        self.key_pressed_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.key_released_cid = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.mouse_pressed_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_press)

    def on_key_press(self, event):

        if (event.key == "a" or event.key == "A") and self.addLineKeyPressedFlag != True:
            self.addLineKeyPressedFlag = True

        if event.key == "g":
            print("Existing lines are (onset, offset, midi:")
            print(self.get_midi_lines())

    def on_key_release(self, event):

        self.addLineKeyPressedFlag = False
        self.newLineParameters = []             # remove incomplete points added to list

    # def on_release(self, event):
    #    self.holdFlag = False

    def on_press(self, event):
        if event.dblclick and (event.xdata and event.ydata):
            if self.addLineKeyPressedFlag:
                if self.newLineParameters == []:
                    self.newLineParameters.append(event.xdata)    # add onset value
                    self.newLineParameters.append(event.ydata)    # add midi value
                else:
                    self.newLineParameters.append(event.xdata)  # add onset value
                    onset = self.newLineParameters[0]
                    offset = self.newLineParameters[2]
                    if self.y_isHz:
                        midi = self.pitch2midi(self.newLineParameters[1], quantizePitch=False)
                    else:
                        midi = self.newLineParameters[1]
                    self.add_midi_line(onset, offset, midi)
                    self.newLineParameters = []
            return

    def on_motion(self, event):
        return

    def add_midi_line(self, onset, offset, midi):
        # creates a midi line using the provided options
        line1 = DraggableHLine(self.fig, self.ax, onset, offset, midi, **self.options)

        self.midi_draggableLines.append(line1)
        self.fig.canvas.updateGeometry()

    def redraw(self):
        self.draw_grid()
        if self.midi_draggableLines:
            for midi_draggableLine in self.midi_draggableLines:
                if midi_draggableLine.get_line():
                    midi_draggableLine.draw()
        self.fig.canvas.draw()

    def draw_grid(self):
        # draws the grid in figure
        for grid in self.horizontal_snap_grid:
            self.ax.axvline(grid, 0, 140)
        self.fig.canvas.draw()

    def get_midi_lines(self):
        # returns a list of midi lines. Each Entry is formatted as (onset, offset, midi value)
        lines = []
        for midi_draggableLine in self.midi_draggableLines:
            onset, offset, midi_val = midi_draggableLine.get_line()
            if onset!=None:
                lines.append((onset, offset, midi_val))

        if lines == []:
            return []
        else:
            return sorted(lines)

    def pitch2midi(self, pitch, quantizePitch=False):
        # converts Hz to Midi number
        midi = []

        if pitch == 0:
            midi = 0
        else:
            if quantizePitch:
                midi = (np.int(np.round(69 + 12 * np.math.log(pitch / 440.0, 2), 0)))
            else:
                midi = 69 + 12 * np.math.log(pitch / 440.0, 2)
        return midi

    def midi2pitch(self, midi):
        return 2**((midi-69)/12.0)*440


class ChromagramCanvas(FigureCanvas):
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
        #       threshold               :   (dB) threshold to remove low energy bins
        #       playable (dflt=False)   :   if true, double click on figure to play audio (disable if midi lines on top)

        self._width = 5
        self._height = 5
        self._dpi = 100

        # Set the x limits of the window
        self.xlim = (0, 20)

        self.filename = []

        # initialize variables used for calculation of spectrogram
        self.audio = []
        self.chromagram = array([])  # Chromagram
        self.timeAxSec = []
        self.fft_size = 1024
        self.frame_size = 1024
        self.hop_size = 256
        self.win_type = "hann"
        self.window = get_window(self.win_type, self.frame_size)
        self.sample_rate = 44100
        self.threshold = []
        self.playable = False

        #   Variables
        self.pitchClasses = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

        # check and set provided options
        self.update_data(**options)

        # create figure and axes for plotting
        self.fig = Figure(figsize=(self._width, self._height), dpi=self._dpi)
        self.ax_chromagram = self.fig.add_subplot(111)

        # Figure Canvas initialization
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

        #   variables used for plotting the moving vertical audio scroll bar
        if self.playable:
            self.fig.canvas.mpl_connect('button_press_event', self.start_stop_play_vline)

        self.play_vline = Line2D([0, 0], [0, 40000], color="r")
        self.ax_chromagram.add_line(self.play_vline)

        self._vline_thread = None  # used for audio playback vline
        self.vline_start = 0
        self.vline_current = 0

        self.vline_move_resolution = .2  # (seconds) amount to move line

        self.is_playing = False

        #initialize figure
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if self.filename:    # if a file name is provided in constructor, plot the spectrogram
            self.set_data_and_plots()

        self.show()

    def get_chromagram_fig(self):
        return self.fig

    def get_chromagram_ax(self):
        return self.ax_chromagram

    def update_data(self, **options):
        if "width" in options:
            self._width = options.get("width")
        if "height" in options:
            self._height = options.get("height")
        if "dpi" in options:
            self._dpi = options.get("dpi")
        if "xlim" in options:
            self.xlim = options.get("xlim")
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

    def load_audio(self):
        # loads the audio
        # apply equal-loudness filter for PredominantPitchMelodia
        loader = es.MonoLoader(filename=self.filename, sampleRate=self.sample_rate)
        self.audio = loader()

    def calc_chromagram(self):

        # save the results in the stft_pool
        self.chromagram = []
        hpcp = es.HPCP(size=12,     # we will need higher resolution for Key estimation
                       referenceFrequency=440,  # assume tuning frequency is 44100.
                       bandPreset=False,
                       weightType='cosine',
                       nonLinear=False,
                       windowSize=1.,
                       sampleRate=self.sample_rate)

        spectrum = es.Spectrum(size=self.fft_size)
        spectral_peaks = es.SpectralPeaks(sampleRate=self.sample_rate)

        for frame in es.FrameGenerator(self.audio, frameSize=self.frame_size,
                                       hopSize=self.hop_size, startFromZero=True):
            frame = array(frame * self.window)
            freqs, mags = spectral_peaks(spectrum(frame))
            chroma = hpcp(freqs, mags)
            self.chromagram.append(chroma)

        self.chromagram = array(self.chromagram)

        self.timeAxSec = np.arange(len(self.chromagram))*self.hop_size/float(self.sample_rate)

    def plot_chromagram(self):
        self.ax_chromagram.cla()
        self.ax_chromagram.set_xlim(*self.xlim)
        y_ax = list(np.arange(12))
        self.ax_chromagram.set_yticks(y_ax)
        self.ax_chromagram.set_yticklabels(self.pitchClasses)
        self.ax_chromagram.pcolormesh(self.timeAxSec, y_ax, self.chromagram.T)
        self.ax_chromagram.set_ylabel("Pitch Class")
        self.fig.canvas.draw()
        return

    def set_data_and_plots(self):
        # loads audio, calculates stft and updates the plot
        self.load_audio()
        self.calc_chromagram()
        self.plot_chromagram()
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
    grid = list(range(21))

    width = 5
    height = 4
    dpi = 100

    app = QtWidgets.QApplication(sys.argv)
    ex = MidiCanvas(horizontal_snap_grid=grid, snapVerticallyFlag=True,
                    snap_offset_flag=True, doubleClickColor="y")

    ex.add_midi_line(19.1, 19.3, 12)

    sys.exit(app.exec_())