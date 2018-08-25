from __future__ import unicode_literals
import sys
import os
import shutil
import glob

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import essentia

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *

from threading import Thread
from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT)

from matplotlib.widgets import RectangleSelector

from midiCanvas import MidiCanvas
from FileManagerGroupBox import FileManagerGroupBox

import numpy as np

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

from spectrogramCanvas import InteractiveSpectrogramCanvas
from basslineSpectrogramCanvas import InteractiveBasslineSpectrogramCanvas
from basslineTranscriber import BasslineTranscriber
from TranscriberCanvas import DrumAndBasslineTranscriberWindow
from drumTranscriber import DrumTranscriber

sys.path.append('../SparsenessSmoothness')
sys.path.append('../MedianFiltering')

from DecomposeSmoothSparse import HPSS as SMSP_HPSS  # smoothness/sparseness based harmonic/percussive source separation

from librosa.core import stft, istft                    # Librosa Version 0.6.0
import librosa.decompose

import essentia.standard as es

from essentia import array

class NavigationToolbar(NavigationToolbar2QT):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Pan', 'Zoom')]

def catch_exceptions(t, val, tb):
    QtWidgets.QMessageBox.critical(None,
                                   "An exception was raised",
                                   "Exception type: {}".format(t))
    old_hook(t, val, tb)


class ApplicationWindow(QtWidgets.QMainWindow):

    StatusBarSignal = pyqtSignal(str)
    FindSeparatedFilesSignal = pyqtSignal()

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.main_widget = QtWidgets.QWidget(self)

        top_level_qt_widget = QtWidgets.QGridLayout(self.main_widget)

        self.file_manager_gbox = FileManagerGroupBox(self.main_widget, group_title="File Manager",
                                                     load_directory_format="../dataset/*/")

        self.mixed_canvas = InteractiveSpectrogramCanvas(parent=self.main_widget, ylim=(20, 200),
                                                         group_title="MIXED SPECTROGRAM")
        self.harmonic_canvas = InteractiveBasslineSpectrogramCanvas(parent=self.main_widget)

        self.percussive_canvas = InteractiveSpectrogramCanvas(parent=self.main_widget,
                                                              group_title="PERCUSSIVE SPECTROGRAM")

        # Initialize analysis values in the comboboxes
        self.mixed_canvas.set_combo_box_current_texts("16384", "2048", "1/16")
        self.harmonic_canvas.set_combo_box_current_texts("16384", "4096", "1/16")
        self.percussive_canvas.set_combo_box_current_texts("16384", "4096", "1/16")

        # button to synchronize the x-y limits of all plots
        self.sync_button = QtWidgets.QPushButton('Synchronize XY Limits')
        self.sync_button.clicked.connect(self.sync_x_y_limits)

        # button to load file into project
        self.load_button = QtWidgets.QPushButton('   Load   ')
        self.load_button.clicked.connect(self.load_mixed_file)


        # separate and analyze buttons and algorithm selection check box
        self.median_checkbox = QtWidgets.QCheckBox('Median Filtering', self)    # check box to separate using median
        self.SMSP_checkbox = QtWidgets.QCheckBox('SMSP NMF', self)              # check box to separate using SMSP
        self.median_checkbox.setChecked(False)
        self.SMSP_checkbox.setChecked(True)
        self.separate_analyze_button = QtWidgets.QPushButton('   Separate and Analyze   ')
        self.separate_analyze_button.clicked.connect(self.separate_and_analyze)
        self.separate_analyze_button.setDisabled(True)  # Next  should be disabled before finding tracks

        # bassline file selector drop down menu combobox
        self.bassline_files_comboBox = QtWidgets.QComboBox(self)    # shows the available separated bassline files
        self.bassline_files_comboBox.setDisabled(True)       # no bassline files available before selecting a mix track
        self.bassline_files_comboBox.currentTextChanged.connect(self.draw_bassline)  # selection callback
        self.harmonic_canvas.main_layout.addWidget(self.bassline_files_comboBox, 4, 0)

        # bassline HarmonicModelAnal
        self.bassline_harmonic_anal_button = QtWidgets.QPushButton('Sinusoidal Analysis')
        self.bassline_harmonic_anal_button.clicked.connect(self.bassline_harmonic_anal)
        self.bassline_harmonic_anal_button.setDisabled(True)  # Next  should be disabled before finding tracks

        # Bassline automatic transcriber
        self.low_pass_checkBox = QtWidgets.QCheckBox('Low Pass', self)
        self.cutoff_lineEdit = QtWidgets.QLineEdit(self)
        self.cutoff_lineEdit.setText("500")
        self.bass_prefix_label = QtWidgets.QLabel(self)
        self.bass_prefix_label.setText("Save w/ Prefix:")
        self.bass_prefix_lineEdit = QtWidgets.QLineEdit(self)
        self.bass_prefix_lineEdit.setText("transcription/bassline_transcription_")
        self.bassline_transcribe_button = QtWidgets.QPushButton('Transcribe')
        self.bassline_transcribe_button.clicked.connect(self.transcribe_bassline)

        # drum file selector drop down menu combobox
        self.drum_files_comboBox = QtWidgets.QComboBox(self)  # shows the available separated bassline files
        self.drum_files_comboBox.setDisabled(True)  # no bassline files available before selecting a mix track
        self.drum_files_comboBox.currentTextChanged.connect(self.draw_drum)  # selection callback
        self.percussive_canvas.main_layout.addWidget(self.drum_files_comboBox, 4, 0)
        self.reestimate_beats_checkbox = QtWidgets.QCheckBox('Estimate Beats (Not Using Segmentation Json)', self)  # check box to separate using SMSP
        self.reestimate_beats_checkbox.setChecked(True)

        # Drum automatic transcriber
        self.drum_prefix_label = QtWidgets.QLabel(self)
        self.drum_prefix_label.setText("Save w/ Prefix:")
        self.drum_prefix_lineEdit = QtWidgets.QLineEdit(self)
        self.drum_prefix_lineEdit.setText("transcription_7Bands/drum_transcription_")
        self.drum_transcribe_button = QtWidgets.QPushButton('Transcribe')
        self.drum_transcribe_button.clicked.connect(self.transcribe_drum)
        
        # organize widget layout
        #           File Manager
        top_level_qt_widget.addWidget(self.file_manager_gbox, 0, 0, 1, 3)
        self.file_manager_gbox.layout.addWidget(self.load_button, 2, 0)      # loads selected file into mixed spgrm
        self.file_manager_gbox.layout.addWidget(self.sync_button, 2, 1)

        #           Mixed Audio Widget
        top_level_qt_widget.addWidget(self.mixed_canvas, 2, 0, 2, 1)
        self.mixed_canvas.main_layout.addWidget(self.median_checkbox, 3, 0, Qt.AlignLeft)     # separates using smsp or median
        self.mixed_canvas.main_layout.addWidget(self.SMSP_checkbox, 3, 0, Qt.AlignRight)  # separates using smsp or median
        self.mixed_canvas.main_layout.addWidget(self.separate_analyze_button, 5, 0)  # separates using smsp or median

        #           Bassline Audio Widget
        top_level_qt_widget.addWidget(self.harmonic_canvas, 2, 1, 2, 1)
        self.harmonic_canvas.main_layout.addWidget(self.bassline_harmonic_anal_button, 5,
                                                   0, Qt.AlignLeft)  # separates using smsp or median
        self.harmonic_canvas.main_layout.addWidget(self.low_pass_checkBox, 5,
                                                   0, Qt.AlignCenter)  # low pass activation
        self.harmonic_canvas.main_layout.addWidget(self.cutoff_lineEdit, 5,
                                                   0, Qt.AlignRight)  # low pass frequency
        top_level_qt_widget.addWidget(self.harmonic_canvas, 2, 1, 2, 1)

        #           Drum Audio Widget
        top_level_qt_widget.addWidget(self.percussive_canvas, 2, 2, 2, 1)
        self.percussive_canvas.main_layout.addWidget(self.reestimate_beats_checkbox, 5, 0)

        #           Add transcription buttons
        top_level_qt_widget.addWidget(self.bass_prefix_label, 4, 1, Qt.AlignLeft)
        top_level_qt_widget.addWidget(self.bass_prefix_lineEdit, 4, 1, Qt.AlignCenter)
        top_level_qt_widget.addWidget(self.bassline_transcribe_button, 4,
                                      1, Qt.AlignRight)  # transcribe button

        top_level_qt_widget.addWidget(self.drum_prefix_label, 4, 2, Qt.AlignLeft)
        top_level_qt_widget.addWidget(self.drum_prefix_lineEdit, 4, 2, Qt.AlignCenter)
        top_level_qt_widget.addWidget(self.drum_transcribe_button, 4,
                                      2, Qt.AlignRight)  # transcribe button

        top_level_qt_widget.setRowStretch(0, 1)
        top_level_qt_widget.setRowStretch(2, 10)


        # Add matplotlib interactive toolbar
        NavToolBar = NavigationToolbar(self.harmonic_canvas.MidiCanvas.fig.canvas, self)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavToolBar)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        # Bassline Transcription Array
        self.bassline_midis = []    #  each element should be formatted as onset, offset, midi val
        # connect signal event handlers
        self.StatusBarSignal.connect(self.StatusBarUpdate)
        self.FindSeparatedFilesSignal.connect(self.find_separated_files)

        #
        self.statusBar().showMessage("", 2000)
        self.resize(1200, 1000)

    def sync_x_y_limits(self):
        # the following lines make sure that all the plots are synchronized at all times
        midi_ax = self.harmonic_canvas.get_midi_ax()
        self.mixed_canvas.share_with_external_ax(midi_ax)
        self.percussive_canvas.share_with_external_ax(midi_ax)

    def transcribe_drum(self):

        drum_audio, sampleRate = self.percussive_canvas.get_audio()
        mX = self.percussive_canvas.get_stft()

        frameSize = int(self.harmonic_canvas.frame_size_comboBox.currentText())
        fftSize = int(self.harmonic_canvas.fft_size_comboBox.currentText())
        hopSize = int(eval(self.harmonic_canvas.hop_size_comboBox.currentText()) * frameSize)

        segmentationJsonFilename = os.path.join(os.path.join(os.path.dirname(self.drum_files_comboBox.currentText()),
                                                             '..'), "segmentation_data.json")

        if self.reestimate_beats_checkbox.checkState():
            segmentationJsonFilename = None
        else:
            segmentationJsonFilename = segmentationJsonFilename

        # ------ Transcribe Drums
        self.StatusBarUpdate("Drum Transcription in Progress")
        AutoDrumLTranscriber = DrumTranscriber(audio=drum_audio,
                                               stft=mX,
                                               frameSize=frameSize,
                                               hopSize=hopSize,
                                               fftSize=fftSize,
                                               onset_method="hfc",
                                               winType="hann",
                                               sampleRate=sampleRate,
                                               onsetframeSize=1024,
                                               onsethopSize=512,
                                               pitchframeSize=1024,
                                               pitchhopSize=512,
                                               minFrequency=60,
                                               maxFrequency=300,
                                               beatDivision=4,
                                               deleteWhereNoOnset=False,
                                               onsetToStartMaxSecs=.1,
                                               postOnsetPecentage=.5,
                                               splitWhereOnset=False,
                                               deleteShortTracks=True,
                                               minFramesofTrack=2,
                                               snapGrid=False,
                                               snapEnd=False,
                                               segmentationJsonFilename=segmentationJsonFilename
                                               )

        drum_analysisResults = AutoDrumLTranscriber.onsets_per_bands
        grid = drum_analysisResults["grid"][0]
        beats = drum_analysisResults["beats"][0]
        onsets = drum_analysisResults["onsets"][0]

        if (beats[0]-(beats[1]-beats[0]))>0:
            beats = np.insert(beats,0, beats[0])

        print("grid", grid)
        print("beats", beats)
        print("onsets", onsets)

        # Create the transcriber pyqt object ( new window will pop-up)
        self.StatusBarUpdate("Opening Transcription Window")
        self.TranscriberWindow = \
            DrumAndBasslineTranscriberWindow(
                bassline_filename=[],
                drum_filename=self.drum_files_comboBox.currentText(),
                grid=grid,
                frame_size=frameSize,
                hop_size=hopSize,
                fft_size=fftSize,
                sample_rate=44100,
                xlim=[0, max(grid)+.5],
                beats=beats,
                bassline_onsets=[],
                midi_tracks=[],
                drum_analysisResults=drum_analysisResults,
                prefix_text = self.drum_prefix_lineEdit.text()
            )

    def transcribe_bassline(self):

        drum_audio, sampleRate = self.percussive_canvas.get_audio()

        mX = self.percussive_canvas.get_stft()

        frameSize = int(self.harmonic_canvas.frame_size_comboBox.currentText())
        fftSize = int(self.harmonic_canvas.fft_size_comboBox.currentText())
        hopSize = int(eval(self.harmonic_canvas.hop_size_comboBox.currentText()) * frameSize)

        segmentationJsonFilename = os.path.join(os.path.join(os.path.dirname(self.drum_files_comboBox.currentText()),
                                                             '..'), "segmentation_data.json")

        if self.reestimate_beats_checkbox.checkState():
            segmentationJsonFilename = None
        else:
            segmentationJsonFilename = segmentationJsonFilename

        # ------ Transcribe Drums
        self.StatusBarUpdate("Drum Transcription in Progress")
        AutoDrumLTranscriber = DrumTranscriber(audio=drum_audio,
                                               stft=mX,
                                               frameSize=frameSize,
                                               hopSize=hopSize,
                                               fftSize=fftSize,
                                               onset_method="hfc",
                                               winType="hann",
                                               sampleRate=sampleRate,
                                               onsetframeSize=1024,
                                               onsethopSize=512,
                                               pitchframeSize=1024,
                                               pitchhopSize=512,
                                               minFrequency=60,
                                               maxFrequency=300,
                                               beatDivision=4,
                                               deleteWhereNoOnset=False,
                                               onsetToStartMaxSecs=.1,
                                               postOnsetPecentage=.5,
                                               splitWhereOnset=False,
                                               deleteShortTracks=True,
                                               minFramesofTrack=2,
                                               snapGrid=True,
                                               snapEnd=True,
                                               segmentationJsonFilename=segmentationJsonFilename
                                               )

        beats, grid, _ = AutoDrumLTranscriber.onsets_broad_band()

        if (beats[0] - (beats[1] - beats[0])) > 0:
            beats = np.insert(beats, 0, beats[0])

        print("grid", grid)
        print("beats", beats)

        # ------ Transcribe Bassline
        bassline_audio, sampleRate = self.harmonic_canvas.get_audio()
        bassline_length = len(bassline_audio)/sampleRate

        # ------ Extend the grid in case the drum loop is not the same length as the bassline
        while True:
            new_grid_line = grid[-1]+(grid[-1]-grid[-2])
            if new_grid_line <= (bassline_length+1):
                grid = np.append(grid, new_grid_line)
            else:
                break

        if self.low_pass_checkBox.checkState():
            print("LOW PASSING")
            bassline_audio = es.LowPass(cutoffFrequency=float(self.cutoff_lineEdit.text()))(bassline_audio)
            bassline_audio = es.LowPass(cutoffFrequency=float(self.cutoff_lineEdit.text()))(bassline_audio)
            bassline_audio = es.LowPass(cutoffFrequency=float(self.cutoff_lineEdit.text()))(bassline_audio)
            bassline_audio = es.LowPass(cutoffFrequency=float(self.cutoff_lineEdit.text()))(bassline_audio)

        self.StatusBarUpdate("Bassline Transcription in Progress")
        AutoBassLTranscriber = BasslineTranscriber(audio=bassline_audio,
                                                   grid=grid,
                                                   frameSize=frameSize,
                                                   hopSize=hopSize,
                                                   fftSize=fftSize,
                                                   onset_method="hfc",
                                                   winType="hann",
                                                   sampleRate=sampleRate,
                                                   onsetframeSize=1024,
                                                   onsethopSize=512,
                                                   pitchframeSize=1024,
                                                   pitchhopSize=512,
                                                   minFrequency=60,
                                                   maxFrequency=300,
                                                   beatDivision=8,
                                                   deleteWhereNoOnset=False,
                                                   onsetToStartMaxSecs=.1,
                                                   postOnsetPecentage=.5,
                                                   splitWhereOnset=False,
                                                   deleteShortTracks=True,
                                                   minFramesofTrack=2,
                                                   snapGrid=False,
                                                   snapEnd=False)

        # midi_tracks, beats, _grid = AutomaticTranscriber.extractorEssentia()    # uses pitchMelodia
        midi_tracks, _, _, bassline_onsets, estQuantizedMIDI, frame_times = AutoBassLTranscriber.onsets_with_pitch  # uses Yin
        print("bassline_onsets", bassline_onsets)

        # remove the first grid location if the distance to next grid line doesnt match with the rest
        if (grid[1]-grid[0])<(grid[2]-grid[1]):
            grid = grid[1:]

        # Update the grid locations in the docked bassline canvas
        self.harmonic_canvas.new_midi_canvas(grid)
        self.harmonic_canvas.update_grid(grid)

        # Create the transcriber pyqt object ( new window will pop-up)
        self.StatusBarUpdate("Opening Transcription Window")
        self.TranscriberWindow = \
            DrumAndBasslineTranscriberWindow(
                bassline_filename=self.bassline_files_comboBox.currentText(),
                drum_filename=[],
                grid=grid,
                frame_size=frameSize,
                hop_size=hopSize,
                fft_size=fftSize,
                sample_rate=44100,
                xlim=[0, max(grid)+.5],
                beats=beats,
                bassline_onsets=bassline_onsets,
                midi_tracks=midi_tracks,
                drum_analysisResults=[],
                prefix_text = self.bass_prefix_lineEdit.text(),
                PYIN_midi = estQuantizedMIDI,
                YIN_times = frame_times
            )

    def bassline_harmonic_anal(self):
        params = dict()
        bassline_audio, params["sampleRate"] = self.harmonic_canvas.get_audio()

        params["frameSize"] = int(self.harmonic_canvas.frame_size_comboBox.currentText())
        params["fftSize"] = int(self.harmonic_canvas.fft_size_comboBox.currentText())
        params["hopSize"] = int(eval(self.harmonic_canvas.hop_size_comboBox.currentText())*params["frameSize"])

        params["fftSize"] = params["hopSize"]

        params["maxFrequency"] = 20000
        params["minFrequency"] = 20

        params["maxnSines"] = 100
        params["magnitudeThreshold"] = -85
        params['minSineDur'] = .05
        params["freqDevOffset"] = 50
        params["freqDevSlope"] = .1

        _, sine_audio, res_audio = self.analysis_synthesis_spr_model_standard(params, bassline_audio)

        #filename = self.harmonic_canvas.
        har_sav_location = self.bassline_files_comboBox.currentText()[:-4].replace("_harmonic", "_spr_sine.wav")
        res_sav_location = self.bassline_files_comboBox.currentText()[:-4].replace("_harmonic", "_spr_res.wav")
        per_sav_location = self.drum_files_comboBox.currentText()[:-4].replace("_percussive", "_with_spr_res.wav")

        drums_audio, sample_rate = self.percussive_canvas.get_audio()

        self.StatusBarSignal.emit("Separating Sinusoidal And Residual Parts")
        Length = min(len(sine_audio), len(res_audio), len(drums_audio))
        drums_audio_plus_residual = drums_audio[-Length:] + res_audio[-Length:]  # add residual to percussive part

        self.StatusBarSignal.emit("Writing Separated Files and the Enhanced Percussion Audio")
        es.MonoWriter(filename=har_sav_location, format="wav")(sine_audio)
        es.MonoWriter(filename=res_sav_location, format="wav")(res_audio)
        es.MonoWriter(filename=per_sav_location, format="wav")(drums_audio_plus_residual)

        self.find_separated_files() # update drop down file lists

    def analysis_synthesis_spr_model_standard(self, params, signal):

        pool = essentia.Pool()
        #   Streaming Algos for Sine Model Analysis
        w = es.Windowing(type="hann")
        fft = es.FFT(size=params['fftSize'])
        smanal = es.SineModelAnal(sampleRate=params['sampleRate'], maxnSines=params['maxnSines'],
                                  magnitudeThreshold=params['magnitudeThreshold'],
                                  freqDevOffset=params['freqDevOffset'], freqDevSlope=params['freqDevSlope'])

        #   Standard Algos for Sine Model Analysis
        smsyn = es.SineModelSynth(sampleRate=params['sampleRate'], fftSize=params['frameSize'],
                                  hopSize=params['hopSize'])
        ifft = es.IFFT(size=params['frameSize'])
        overlSine = es.OverlapAdd(frameSize=params['frameSize'], hopSize=params['hopSize'],
                                  gain=1. / params['frameSize'])
        overlres = es.OverlapAdd(frameSize=params['frameSize'], hopSize=params['hopSize'],
                                 gain=1. / params['frameSize'])

        fft_original = []

        # analysis
        for frame in es.FrameGenerator(signal, frameSize=params["frameSize"], hopSize=params["hopSize"]):
            frame_fft = fft(w(frame))
            fft_original.append(frame_fft)
            freqs, mags, phases = smanal(frame_fft)
            pool.add("frequencies", freqs)
            pool.add("magnitudes", mags)
            pool.add("phases", phases)

        # remove short tracks
        minFrames = int(params['minSineDur'] * params['sampleRate'] / params['hopSize'])
        pool = self.cleaningSineTracks(pool, minFrames)

        # synthesis
        sineTracksAudio = np.array([])
        resTracksAudio = np.array([])
        for frame_ix, _ in enumerate(pool["frequencies"]):
            sine_frame_fft = smsyn(pool["magnitudes"][frame_ix],
                                   pool["frequencies"][frame_ix],
                                   pool["phases"][frame_ix])
            res_frame_fft = fft_original[frame_ix]-sine_frame_fft
            sine_outframe = overlSine(ifft(sine_frame_fft))
            sineTracksAudio = np.append(sineTracksAudio, sine_outframe)
            res_outframe = overlres(ifft(res_frame_fft))
            resTracksAudio = np.append(resTracksAudio, res_outframe)

        sineTracksAudio = sineTracksAudio.flatten()[-len(signal):]
        resTracksAudio = resTracksAudio.flatten()[-len(signal):]

        #print("len signal", len(signal), "len res", len(resTracksAudio))
        return essentia.array(signal), essentia.array(sineTracksAudio), essentia.array(resTracksAudio)

    def cleaningSineTracks(self, pool, minFrames):
        """
        Cleans the sine tracks identified based on the minimum number of frames identified
        reference: https://github.com/MTG/essentia/blob/b5b46f80d80058603a525af36cbf7069c17c3df9/
        test/src/unittests/synthesis/test_sinemodel_streaming.py

        :param pool: must contain pool["magnitudes"], pool["frequencies"] and pool["phases"]
        :param minFrames: minimum number of frames required for a sine track to be valid
        :return: cleaned up pool
        """

        freqsTotal = pool["frequencies"]
        nFrames = freqsTotal.shape[0]
        begTrack = 0
        freqsClean = freqsTotal.copy()

        if (nFrames > 0):

            f = 0
            nTracks = freqsTotal.shape[1]  # we assume all frames have a fix number of tracks

            for t in range(nTracks):

                f = 0
                begTrack = f

                while (f < nFrames - 1):

                    # // check if f is begin of track
                    if (freqsClean[f][t] <= 0 and freqsClean[f + 1][t] > 0):
                        begTrack = f + 1

                    # clean track if shorter than min duration
                    if ((freqsClean[f][t] > 0 and freqsClean[f + 1][t] <= 0) and ((f - begTrack) < minFrames)):
                        for i in range(begTrack, f + 1):
                            freqsClean[i][t] = 0

                    f += 1

        cleaned_pool = essentia.Pool()

        for frame_ix, originalTracks in enumerate(freqsTotal):
            freqs = []
            mags = []
            phases = []
            for track_ix, freqTrack in enumerate(originalTracks):
                if freqTrack in freqsClean[frame_ix]:
                    freqs.append(pool["frequencies"][frame_ix][track_ix])
                    mags.append(pool["magnitudes"][frame_ix][track_ix])
                    phases.append(pool["phases"][frame_ix][track_ix])
                else:
                    freqs.append(0)
                    mags.append(0)
                    phases.append(0)
            cleaned_pool.add("frequencies", essentia.array(freqs))
            cleaned_pool.add("magnitudes", essentia.array(mags))
            cleaned_pool.add("phases", essentia.array(phases))

        return cleaned_pool

    # converts audio frames to a single array
    def frames_to_audio(self, frames):
        frames = essentia.array(frames)
        audio = frames.flatten()
        return audio

    def add_frames(self, frames, hopSize):
        nFrames = len(frames)
        frameSize = len(frames[0])
        audio = np.array([])
        for i, frame in enumerate(frames):
            if i==0:
                audio = frame
            else:
                overlapRegion = frames[i-1][hopSize:]+frame[:-hopSize]
                audio = np.append(audio, overlapRegion)
                audio = np.append(audio, frame[-hopSize:])
        return audio

    def StatusBarUpdate(self, message):
        self.statusBar().showMessage(message)

    def find_separated_files(self):
        self.find_bassline_files()
        self.find_drum_files()

    def load_mixed_file(self):
        self.statusBar().showMessage(self.file_manager_gbox.get_current_filename(), 2000)
        self.mixed_canvas.set_filename(self.file_manager_gbox.get_current_filename())
        self.separate_analyze_button.setDisabled(False)
        self.find_bassline_files()
        self.find_drum_files()
        self.repaint()
        return

    def separate_and_analyze(self):
        thread = Thread(target=self.separate_and_analyze_function)
        thread.start()

    def separate_and_analyze_function(self):
        # Separation using Smoothness/Sparseness
        frameSize = self.mixed_canvas.frame_size_comboBox.currentText()
        frameSize = int(frameSize)
        hopSize = self.mixed_canvas.hop_size_comboBox.currentText()
        hopSize = int(eval(hopSize) * frameSize)
        fftSize = self.mixed_canvas.fft_size_comboBox.currentText()
        fftSize = int(fftSize)

        x, sampleRate = self.mixed_canvas.get_audio()
        file_directory = os.path.dirname(self.mixed_canvas.get_filename())

        # File name format Algo_FFTSize_frameSize_hopSize
        SMSP_filename_prefix = "SMSP_"+str(fftSize)+"_"+str(frameSize)+"_"+str(hopSize)
        median_filename_prefix = "median_"+str(fftSize)+"_"+str(frameSize)+"_"+str(hopSize)

        if not x == []:
            if self.SMSP_checkbox.checkState():
                self.StatusBarSignal.emit("Separating Using Smoothness/Sparseness NMF Algorithm")

                #   separate using Smoothness/Sparseness
                hpss = SMSP_HPSS(
                    np.array(x),
                    directory=file_directory,
                    filename=SMSP_filename_prefix,
                    format="wav",
                    beta=1.5,
                    frameSize=frameSize,
                    hopSize=hopSize,
                    fftSize=fftSize,
                    Rp=150,
                    Rh=150,
                    K_SSM=.2,  # Percussive Spectral Smoothness
                    K_TSP=.1,  # Percussive Temporal Smoothness
                    K_SSP=.1,  # Harmonic Spectral Smoothness
                    K_TSM=.2,  # Harmonic Temporal Smoothness
                )

                maxIter = 100

                for i in range(int(maxIter)):
                    self.StatusBarSignal.emit("Iteration %i out of %i" % (i + 1, maxIter))
                    hpss.next_iteration()

                hpss.create_masks()
                hpss.spectral_to_temporal_using_masks()

                hpss.save_separated_audiofiles()

                shutil.move(os.path.join(file_directory, SMSP_filename_prefix+"_harmonic.wav"),
                            os.path.join(os.path.join(file_directory, "harmonic"),
                                         SMSP_filename_prefix+"_harmonic.wav"))

                shutil.move(os.path.join(file_directory, SMSP_filename_prefix+"_percussive.wav"),
                            os.path.join(os.path.join(file_directory, "percussive"),
                                         SMSP_filename_prefix+"_percussive.wav"))

                # self.StatusBarSignal.emit("Finished Separating (SMSP)")

            if self.median_checkbox.checkState():
                self.StatusBarSignal.emit("Separating Using Median Filtering Algorithm")
                # Separation using median filtering
                _stft = stft(x, n_fft=fftSize, hop_length=hopSize, win_length=frameSize, window="hann")

                X_H, X_P = librosa.decompose.hpss(_stft, kernel_size=150)  # Get harmonic and percussive stfts

                x_h = istft(X_H, hop_length=hopSize, win_length=frameSize)  # Convert stfts to time domain signals
                x_p = istft(X_P, hop_length=hopSize, win_length=frameSize)

                MonoWriter = es.MonoWriter(sampleRate=44100, format="wav")  # Write to file
                MonoWriter.configure(filename=
                                     os.path.join(os.path.join(file_directory, "percussive"),
                                                  median_filename_prefix+"_percussive.wav"))
                MonoWriter(array(x_p))

                MonoWriter = es.MonoWriter(sampleRate=44100, format="wav")  # Write to file
                MonoWriter.configure(filename=
                                     os.path.join(os.path.join(file_directory, "harmonic"),
                                                  median_filename_prefix+"_harmonic.wav"))
                MonoWriter(array(x_h))

                # self.StatusBarSignal.emit("Finished Separating (Median Filtering)")

            self.FindSeparatedFilesSignal.emit()

        return

    def find_bassline_files(self):
        mix_track_folder = os.path.dirname(self.file_manager_gbox.get_current_filename())
        bassline_folder = os.path.join(mix_track_folder, "harmonic")
        bassline_files = glob.glob(os.path.join(bassline_folder, "*.mp3"))
        wavs = glob.glob(os.path.join(bassline_folder, "*.wav"))
        if wavs:
            for wav in wavs:
                bassline_files.append(wav)

        # disable combo_box for selecting bassline file
        self.bassline_files_comboBox.setDisabled(True)
        self.bassline_harmonic_anal_button.setDisabled(True)

        # remove all items from the combobox
        while self.bassline_files_comboBox.count()>0:
            self.bassline_files_comboBox.removeItem(0)

        # if there are bassline files available, add them to drop down menu
        if bassline_files:
            self.bassline_files_comboBox.setDisabled(False)
            self.bassline_harmonic_anal_button.setDisabled(False)
            for bassline_file in bassline_files:
                #print(bassline_file)
                self.bassline_files_comboBox.addItem(bassline_file)
        else:
            self.harmonic_canvas.clear_all_plots()
        self.repaint()

    def draw_bassline(self):
        bassline_file = self.bassline_files_comboBox.currentText()
        self.harmonic_canvas.set_filename(bassline_file)
        self.repaint()

    def find_drum_files(self):
        mix_track_folder = os.path.dirname(self.file_manager_gbox.get_current_filename())
        drum_folder = os.path.join(mix_track_folder, "percussive")
        drum_files = glob.glob(os.path.join(drum_folder, "*.mp3"))
        wavs = glob.glob(os.path.join(drum_folder, "*.wav"))
        if wavs:
            for wav in wavs:
                drum_files.append(wav)

        # disable combo_box for selecting bassline file
        self.drum_files_comboBox.setDisabled(True)

        # remove all items from the combobox
        while self.drum_files_comboBox.count()>0:
            self.drum_files_comboBox.removeItem(0)

        # if there are bassline files available, add them to drop down menu
        if drum_files:
            self.drum_files_comboBox.setDisabled(False)
            for drum_file in drum_files:
                #print(drum_file)
                self.drum_files_comboBox.addItem(drum_file)
        else:
            self.percussive_canvas.clear_all_plots()

        self.repaint()

    def draw_drum(self):
        drum_file = self.drum_files_comboBox.currentText()
        self.percussive_canvas.set_filename(drum_file)
        self.repaint()


if __name__ == '__main__':
    grid = list(range(21))

    app = QtWidgets.QApplication(sys.argv)

    aw = ApplicationWindow()
    aw.setWindowTitle("%s" % progname)

    old_hook = sys.excepthook
    sys.excepthook = catch_exceptions

    aw.show()
    sys.exit(app.exec_())
