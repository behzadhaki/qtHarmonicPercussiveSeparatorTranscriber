import numpy as np
import essentia.standard as es
from essentia import Pool, array
from copy import deepcopy
import json

class DrumTranscriber:
    def __init__(self, **options):
        self.audio = None
        self.stft = None

        self.frameSize = 2048
        self.hopSize = 512
        self.winType = "hann"
        self.fftSize = 4096
        self.sampleRate = 44100
        self.minFrequency = 50
        self.maxFrequency = 350
        self.quantizePitch = True
        self.guessUnvoiced = True

        self.onsetMethod = "complex"
        self.onsetFrameSize = 4096
        self.onsetHopSize = 256
        self.beatDivision = 4
        self.deleteWhereNoOnset = True
        self.onsetToStartMaxSecs = .1
        self.postOnsetPercentage = .5
        self.splitWhereOnset = True
        self.deleteShortTracks = True
        self.snapBeginning = True
        self.snapEnd = True
        self.minFramesofTrack = 1
        self.snapGrid = True

        self.analysisResults = None
        self.segmentationJsonFilename = None    # this json contains beat informations --> if not available beats are detected again
        self.updateOptions(**options)

    def updateOptions(self, **options):
        if "audio" in options:
            self.audio = options.get("audio")
        if "stft" in options:
            self.stft = options.get("stft")
        if "frameSize" in options:
            self.frameSize = options.get("frameSize")
        if "hopSize" in options:
            self.hopSize = options.get("hopSize")
        if "winType" in options:
            self.winType = options.get("winType")
        if "fftSize" in options:
            self.fftSize = options.get("fftSize")
        if "sampleRate" in options:
            self.sampleRate = options.get("sampleRate")
        if "minFrequency" in options:
            self.minFrequency = options.get("minFrequency")
        if "maxFrequency" in options:
            self.maxFrequency = options.get("maxFrequency")
        if "quantizePitch" in options:
            self.quantizePitch = options.get("quantizePitch")
        if "guessUnvoiced" in options:
            self.guessUnvoiced = options.get("guessUnvoiced")
        if "onsetFrameSize" in options:
            self.onsetFrameSize = options.get("onsetFrameSize")
        if "onsetHopSize" in options:
            self.onsetHopSize = options.get("onsetHopSize")
        if "beatDivision" in options:
            self.beatDivision = options.get("beatDivision")
        if "deleteWhereNoOnset" in options:
            self.deleteWhereNoOnset = options.get("deleteWhereNoOnset")
        if "onsetToStartMaxSecs" in options:
            self.onsetToStartMaxSecs = options.get("onsetToStartMaxSecs")
        if "postOnsetPercentage" in options:
            self.postOnsetPercentage = options.get("postOnsetPercentage")
        if "splitWhereOnset" in options:
            self.splitWhereOnset = options.get("splitWhereOnset")
        if "deleteShortTracks" in options:
            self.deleteShortTracks = options.get("deleteShortTracks")
        if "snapBeginning" in options:
            self.snapBeginning = options.get("snapBeginning")
        if "snapEnd" in options:
            self.snapEnd = options.get("snapEnd")
        if "minFramesofTrack" in options:
            self.minFramesofTrack = options.get("minFramesofTrack")
        if "snapGrid" in options:
            self.snapGrid = options.get("snapGrid")
        if "segmentationJsonFilename" in options:
            self.segmentationJsonFilename = options.get("segmentationJsonFilename")

    def get_analysisResults(self):
        return self.analysisResults

    def onsets_broad_band(self):
        # uses Yin to calculate midi tracks

        # Find the onset locations
        onsets = sorted(self.get_onsets())
        print("onsets", onsets)

        # Find the beat locations
        if self.segmentationJsonFilename:
            with open(self.segmentationJsonFilename, 'r') as f:
                segmentationDict = json.load(f)["percussive"]
            length_s = float(segmentationDict["length_s"])
            length_bar = int(segmentationDict["length_bar"])
            beats = np.arange(length_bar*4+1)/(length_bar*4)*length_s
            print("BEATS using Json are: ", beats)
        else:
            beats = sorted(self.getBeats())

        # Create the Grid
        grid = sorted(self.createGrid(beats, self.beatDivision))
        print("grid using Json are: ", grid)

        grid_res_in_seconds = grid[1]-grid[0]

        return beats, grid, onsets

    @property
    def onsets_per_bands(self):
        '''
        Performs a band scale onset analysis of the drums
        :return: analysisResults:   an essentia pool containing the bandbased and broadband analysis of the drums
        '''

        # Pool to save results
        analysisResults = Pool()

        # save audio in pool
        analysisResults.add("audio", self.audio)
        drum_length = len(self.audio)/self.sampleRate

        # create grid
        beats, grid, onsets = self.onsets_broad_band()
        grid = array(grid)
        grid = grid[grid<=drum_length]
        analysisResults.add("beats", array(beats))
        analysisResults.add("grid", grid)
        analysisResults.add("onsets", array(onsets))
        grid_res = grid[1]-grid[0]

        # create filter specs (band band band pass filters)
        # ref: http://essentia.upf.edu/documentation/reference/streaming_bandBands.html

        '''
        f0s = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0,
                       630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
                       2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0,
                       6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 20500.0])


        f1s = np.array([50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0,
                       770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
                       2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0,
                       6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 20500.0, 27000.0])

        '''
        #http://www.music.mcgill.ca/~ich/classes/mumt614/similarity/herrera02automatic.pdf
        f0s = np.array([40., 70., 130., 160., 300., 5000., 7000., 10000.])
        f1s = np.array([70., 110., 145., 190., 400., 7000., 10000., 15000.])


        bandwidths = f1s - f0s
        cutoffFrequencies = (f0s + f1s) / 2.

        analysisResults.add("x_time", array(np.arange(len(self.audio))/self.sampleRate))
        analysisResults.add("f0s", array(f0s))
        analysisResults.add("f1s", array(f1s))
        analysisResults.add("bandwidths", array(bandwidths))
        analysisResults.add("cutoffFrequencies", array(cutoffFrequencies))

        # matrix of onsets: dimension 1: freq band dimension 2: onsets snapped to grid (1 where onset, 0 where no onset)
        drum_onsets_quantized = []

        # matrix of energies: dimension 1: freq band dimension 2: energies where 1 in drum_onsets_quantized
        drum_onset_energies_quantized = []

        # filter and find onsets
        for ix, f0 in enumerate(f0s):
            # Create band pass filters and filter the signal
            print("band", str(ix), "is being calculated")
            BPF = es.BandPass(bandwidth=bandwidths[ix],
                              cutoffFrequency=cutoffFrequencies[ix],
                              sampleRate=self.sampleRate)

            signal = BPF(array(self.audio))
            onsets = self.get_onsets(_audio=signal)
            analysisResults.add("audio_band_fc_"+str(cutoffFrequencies[ix]), signal)

            #calculate energy of each onset (starting grid_res/4 before to after)
            energies = []
            EnergyEstimator = es.Energy()
            #maxEnergy = EnergyEstimator(array(np.hanning(int(grid_res/2.0*self.sampleRate)) *
            #                                  np.random(int(grid_res/2.0*self.sampleRate))))   # rough estimate

            '''
            # this part calculates energy within a small windowed frame around the onset
            max_Energy = 0
            for onset_ix, onset in enumerate(onsets):
                #ix0 = int(max(((onset - grid_res/16)*self.sampleRate), 0))
                #ix1 = int(min(((onset + grid_res/5.33)*self.sampleRate), len(signal)-1))
                ix0 = int(max((onset*self.sampleRate), 0))
                ix1 = int(max(ix0 + 512, len(signal)-2))
                sig = signal[ix0:ix1]
                sig = np.append(sig[::-1], sig[1:])
                if len(sig)>=.01*44100:
                    window = es.Windowing(size=int(len(sig)))
                    energies.append(EnergyEstimator(window(sig)))
                else:
                    onsets = np.delete(onsets,onset_ix)
            '''

            # this part calculates energy from one onset to half grid
            max_Energy = 0
            for onset_ix, onset in enumerate(onsets):
                # ix0 = int(max(((onset - grid_res/16)*self.sampleRate), 0))
                # ix1 = int(min(((onset + grid_res/5.33)*self.sampleRate), len(signal)-1))
                ix0 = int(max((onset * self.sampleRate), 0))
                ix1 = int(max((onset * self.sampleRate+grid_res/2), len(signal) - 2))
                sig = signal[ix0:ix1]
                sig = np.append(sig[::-1], sig[1:])
                if len(sig) >= .01 * 44100:
                    #window = es.Windowing(size=int(len(sig)))
                    energies.append(EnergyEstimator(sig))
                else:
                    onsets = np.delete(onsets, onset_ix)

            max_Energy = max(max_Energy, max(np.array(energies)))

            analysisResults.add("onsets_band_fc_"+str(cutoffFrequencies[ix]), onsets)

            analysisResults.add("energies_band_fc_" + str(cutoffFrequencies[ix]), energies)
            quantized_onset_array_in_band, quantized_energy_array_in_band = self.quantize_onsets(onsets, energies, grid)

            drum_onsets_quantized.append(quantized_onset_array_in_band)
            drum_onset_energies_quantized.append(quantized_energy_array_in_band)

        analysisResults.add("onsets_quantized_matrix", array(np.array(drum_onsets_quantized)))
        analysisResults.add("energies_quantized_matrix", array(np.array(drum_onset_energies_quantized/max_Energy)))

        for ix, f0 in enumerate(f0s):
            analysisResults.add("normalized_energies_band_fc_" + str(cutoffFrequencies[ix]),
                                analysisResults["energies_band_fc_" + str(cutoffFrequencies[ix])][0]/max_Energy)

        return analysisResults

    def get_onsets(self, _audio=[]):

        if _audio!=[]:
            audio = _audio
        else:
            audio = self.audio

        W = es.Windowing(type=self.winType)
        c2p = es.CartesianToPolar()
        fft = es.FFT()
        onsetDetection = es.OnsetDetection(method=self.onsetMethod, sampleRate=44100)
        onsets = es.Onsets(alpha=.2)
        # onsetIndex = []
        pool = Pool()

        for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512):
            mag, phase, = c2p(fft(W(frame)))
            onsetDetection.configure(method=self.onsetMethod)
            onsetFunction = onsetDetection(mag, phase)
            pool.add("onsetFunction", onsetFunction)

        DetectedOnsetsArray = onsets([pool["onsetFunction"]], [1])

        return DetectedOnsetsArray

    def getBeats(self):
        #BeatTracker = es.BeatTrackerMultiFeature()
        #beats, _ = BeatTracker(self.audio)
        BeatTracker = es.BeatTrackerDegara(minTempo=90, maxTempo=130)
        beats= BeatTracker(self.audio)
        if beats!=[]:
            if (beats[0]-(beats[1]-beats[0]))>0:
                beats = np.insert(beats, 0, (beats[0]-(beats[1]-beats[0])))
            if (beats[-1]+(beats[1]-beats[0]))>0:
                beats = np.insert(beats, -1, (beats[-1]+(beats[1]-beats[0])))
            elif np.abs(beats[-1]+(beats[1]-beats[0]))<=((beats[1]-beats[0])/4):  #insert beat at 0 if beat goes slightly before
                beats = np.insert(beats, -1, 0)

        return beats

    def snap_onset_to_grid(self, onset, grid):
        # quantize onset to grid locations
        _onset = deepcopy(onset)
        _onset, idx = self.find_nearest(grid, _onset)
        return _onset, idx

    def quantize_onsets(self, onsets, energies, grid):
        # returns an array of onsets based on the grid location
        # also returns an array corresponding to onsets containing the energies at onsets
        # i.e. creates an array of same size as grid, puts 1 if theres an onset, otherwize 0
        #      also creates a similar array containing the energies at onsets

        quantized_onset_array = np.zeros_like(grid)
        quantized_energy_array = np.zeros_like(grid)

        print("onsets: ", onsets)
        print("energies: ", energies)

        for ix, onset in enumerate(onsets):
            _onset, idx = self.snap_onset_to_grid(onset, grid)
            quantized_onset_array[idx] = 1
            quantized_energy_array[idx] += energies[ix]

        return quantized_onset_array, quantized_energy_array

    def find_nearest(self, array, value):
        # returns the closest entry in array to value and also the corresponding index in array
        idx = (np.abs(np.array(array) - value)).argmin()
        return array[idx], idx

    def splitAtOnsets(self, timeStamps, MIDIGroups, onsets):  # Only for pitch quantized groups ---> Not Used
        timeStampGroups = deepcopy(timeStamps)
        newTimeStamps = []
        newMIDIGroups = []

        for i in range(len(timeStampGroups)):
            timeStampGroup = timeStampGroups[i]
            if (timeStampGroup[0] != timeStampGroup[1]):
                onsetsInGroup = list(onset for onset in onsets if timeStampGroup[0] < onset < timeStampGroup[1])

                timeStampGroup.extend(onsetsInGroup)
                timeStampGroup = sorted(timeStampGroup)
                timeStampGroup = np.unique(timeStampGroup)

            for k in range(len(timeStampGroup) - 1):
                newTimeStamps.append([timeStampGroup[k], timeStampGroup[k + 1]])
                newMIDIGroups.append(MIDIGroups[i])

        return newTimeStamps, newMIDIGroups

    def createGrid(self, beats, beatDivision):
        # beatDivision needs to be int > 1
        # endTime is end of the file or can also be where griding stops
        if beatDivision < 1:
            beatDivision = 1
        else:
            beatDivision = int(beatDivision)

        if len(beats) == 1:
            # print("can't create grid with only a beat marker")
            return []

        gridRes = (beats[1] - beats[0]) / float(beatDivision)  # resolution of grid in ms

        grid = np.array([])

        if beatDivision == 1:
            grid = beats
        else:

            grid = np.array(beats[0]) # Grid starts at beat 0
            for k in range(1, 2*beatDivision + 1):  # create the possibly missing grids before beat 0
                if (beats[0] - k * gridRes) >= 0:
                    grid=np.append(beats[0] - k * gridRes, grid)

            for i in range(len(beats)):
                if not i == len(beats) - 1:
                    gridRes = (beats[i + 1] - beats[i]) / float(beatDivision)  # resolution of grid in ms
                for k in range(beatDivision):
                    grid=np.append(grid, beats[i] + k * gridRes)

            grid = np.append(grid, beats[-1])

            grid = np.sort(np.unique(grid))
            grid = grid[grid<=np.max(beats)]

        return grid

    def existingMIDIValues(self, MIDIGroups):
        # returns just the midi values extracted (sorted with no duplicates)
        existingMIDIs = np.array([])

        for MIDIGroup in MIDIGroups:
            existingMIDIs = np.append(existingMIDIs, MIDIGroup[0])

        existingMIDIs = np.unique(existingMIDIs).tolist()

        return sorted(existingMIDIs)
