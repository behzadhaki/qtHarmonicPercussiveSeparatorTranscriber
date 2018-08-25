import numpy as np
import essentia.standard as es
from essentia import Pool, array
from copy import deepcopy


class BasslineTranscriber:
    def __init__(self, **options):
        self.audio = None
        self.stft = None

        self.grid = None

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

        self.updateOptions(**options)

    def updateOptions(self, **options):
        if "audio" in options:
            self.audio = options.get("audio")
        if "stft" in options:
            self.stft = options.get("stft")
        if "grid" in options:
            self.grid = options.get("grid")
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

        self.analysisResults = self.extractorEssentia()

    def get_analysisResults(self):
        return self.analysisResults

    @property
    def onsets_with_pitch(self):
        # uses Yin to calculate midi tracks

        # Find the Predominant Pitch in every frame
        #estPitch, _ = self.get_predominant_pitch()
        estPitch, estPitchConfidence, frame_times = self.get_Yin_Pitch()
        print("estPitch", estPitch)
        print("estPitchConfidence", estPitchConfidence)
        for ix, confidence in enumerate(estPitchConfidence):
            if confidence <= 0.25:
                estPitch[ix] = 0

        estQuantizedMIDI = array(self.pitch2midi(estPitch, quantizePitch=True))

        # print("estPitch are ", estPitch)
        # print("estQuantizedMIDI are ", estQuantizedMIDI)

        # Find the onset locations
        onsets = sorted(self.get_onsets())
        print("onsets", onsets)

        # Find the beat locations
        beats = sorted(self.getBeats())

        # Create the Grid
        if not (self.grid is None):
            grid = self.grid
        else:
            grid = sorted(self.createGrid(beats, self.beatDivision))
        grid_res_in_seconds = grid[1]-grid[0]

        # Create Midi Tracks (each entry: (onset, offset, MIDI value))
        midi_tracks = []
        number_of_onsets = len(onsets)
        frame_times = np.array(frame_times)

        onsets = np.array(onsets)

        #print("frame_times are", frame_times)
        for ix, onset in enumerate(onsets[onsets<=frame_times[-1]]):
            # get pitch value
            if ix<(number_of_onsets-1):
                offset = onsets[ix+1]
            else:
                offset = frame_times[-1]

            print(np.where(frame_times >= onset))
            if np.where(frame_times >= onset)[0]!=[]:
                frame_begin = np.where(frame_times >= onset)[0][0]
                if np.where(frame_times >= offset)[0]!=[]:
                    frame_end = np.where(frame_times >= offset)[0][0]
                else:
                    frame_end = len(frame_times)-1
                # print("frame_begin, frame_end",  frame_begin, frame_end)
                non_zero_pitchs = estQuantizedMIDI[frame_begin:frame_end]
                # print("non_zero_pitchs are ", non_zero_pitchs)
                midi = np.median(non_zero_pitchs)
                # print("midi ", midi)

                midi_tracks.append([onset, offset, midi])
            else:
                continue
        #print("midi_tracks are ", midi_tracks)

        return midi_tracks, beats, grid, onsets, estQuantizedMIDI, frame_times

    def get_Yin_Pitch(self):
        if self.audio != []:
            pitchDetect = es.PitchYin(frameSize=self.frameSize,
                                      sampleRate=self.sampleRate)
            estPitch = []
            pitchConfidence = []
            frame_times = []

            counter = 0
            for frame in es.FrameGenerator(self.audio, frameSize=self.frameSize, hopSize=self.hopSize):
                f, conf = pitchDetect(frame)
                estPitch += [f]
                pitchConfidence += [conf]
                frame_times.append(counter*self.hopSize/self.sampleRate)
                counter+=1

            return np.array(estPitch), pitchConfidence, frame_times
        else:
            return None, None, None


    def extractorEssentia(self):

        # create the results pool
        analysisResults = dict()

        # Find the Predominant Pitch in every frames
        estPitch, _ = self.get_predominant_pitch()

        # convert pitch to MIDI
        estMIDI = self.pitch2midi(estPitch, quantizePitch=True)
        estTime = array(np.arange(len(estMIDI)) * self.hopSize / float(self.sampleRate))
        analysisResults["estTime"] = estTime
        # Find the onset locations
        onsets = self.get_onsets()
        analysisResults["onsets"] = onsets

        # Convert quantized/unquantized midis to essentia arrays
        estUnquantizedMIDI = array(self.pitch2midi(estPitch, quantizePitch=False))
        estQuantizedMIDI = array(self.pitch2midi(estPitch, quantizePitch=True))
        analysisResults["estUnquantizedMIDI"] = estUnquantizedMIDI
        analysisResults["estQuantizedMIDI"] = estQuantizedMIDI

        # Calculate beats using onsets and define the Grid 
        beats = sorted(self.getBeats())
        grid = sorted(self.createGrid(beats, self.beatDivision))
        analysisResults["beats"] = beats
        analysisResults["grid"] = grid

        # Split the quantized/unquantized MIDI arrays into seperate groups
        _, MidiTimesQuantized, QuantizedMIDIgroups = self.splitQuantizedMIDI(estQuantizedMIDI,
                                                                             minFrame=0)

        _, MidiTimesUnquantized, UnquantizedMIDIgroups = self.splitUnquantizedMIDI(estQuantizedMIDI,
                                                                                   estUnquantizedMIDI,
                                                                                   minFrame=0)

        analysisResults["PitchQuantized.TimeStamps.Original"] = MidiTimesQuantized
        analysisResults["PitchQuantized.MIDIgroups.Original"] = QuantizedMIDIgroups
        analysisResults["PitchUnquantized.MidiTimes.Original"] = MidiTimesUnquantized  # just for plotting
        analysisResults["PitchUnquantized.MIDIgroups.Original"] = UnquantizedMIDIgroups  # just for plotting

        # Remove if no onset in the vicinity of the start of the track
        if self.deleteWhereNoOnset:
            MIDITrackTimes, MIDITracks = self.RemoveTracksAwayFromOnsets(MidiTimesQuantized,
                                                                         QuantizedMIDIgroups,
                                                                         onsets,
                                                                         minLength=self.onsetToStartMaxSecs)
        else:
            MIDITrackTimes, MIDITracks = MidiTimesQuantized, QuantizedMIDIgroups

        analysisResults["PitchQuantized.TimeStamps.CloseToOnsets"] = MIDITrackTimes
        analysisResults["PitchQuantized.MIDIGroups.CloseToOnsets"] = MIDITracks

        # Split where onset appears within a track
        if self.splitWhereOnset:
            onsetSplittedTimeStampsQuantized, onsetSplittedQuantizedMIDIGroups = self.splitAtOnsets(MIDITrackTimes,
                                                                                                    MIDITracks,
                                                                                                    onsets)
        else:
            onsetSplittedTimeStampsQuantized, onsetSplittedQuantizedMIDIGroups = MIDITrackTimes, MIDITracks

        analysisResults["PitchQuantized.TimeStamps.onsetSplitted"] = onsetSplittedTimeStampsQuantized
        analysisResults["PitchQuantized.MIDIGroups.onsetSplitted"] = onsetSplittedQuantizedMIDIGroups

        # delete short tracks
        if self.deleteShortTracks:
            MIDITrackTimesSplitted, MIDITracksSplitted = self.RemoveShortTracks(onsetSplittedTimeStampsQuantized,
                                                                                onsetSplittedQuantizedMIDIGroups,
                                                                                onsets,
                                                                                minLength=self.minFramesofTrack)

        else:
            MIDITrackTimesSplitted, MIDITracksSplitted = onsetSplittedTimeStampsQuantized,\
                                                         onsetSplittedQuantizedMIDIGroups

        analysisResults["PitchQuantized.TimeStamps.removedShortTracks"] = MIDITrackTimesSplitted
        analysisResults["PitchQuantized.MIDIGroups.removedShortTracks"] = MIDITracksSplitted

        # Time Quantize the Tracks (snap beginning and end to grid)
        if self.snapGrid:
            MIDITrackTimesSnapped, MIDITracksSnapped = self.SnapToGrid(MIDITrackTimesSplitted,
                                                                       MIDITracksSplitted,
                                                                       grid)
        else:
            MIDITrackTimesSnapped, MIDITracksSnapped = MIDITrackTimesSplitted, MIDITracksSplitted

        analysisResults["Time&PitchQuantized.TimeStamps"] = MIDITrackTimesSnapped
        analysisResults["Time&PitchQuantized.MIDIGroups"] = MIDITracksSnapped

        MIDITrackTimesSnapped = analysisResults["Time&PitchQuantized.TimeStamps"]
        MIDITracksSnapped = analysisResults["Time&PitchQuantized.MIDIGroups"]

        # print ("MIDITrackTimesSnapped", MIDITrackTimesSnapped)
        # print("MIDITracksSnapped", MIDITracksSnapped)

        midi_tracks = []
        for ix, MIDITrackTimeSnapped in enumerate(MIDITrackTimesSnapped):
            (onset, offset, midi) = (MIDITrackTimeSnapped[0], MIDITrackTimeSnapped[1], MIDITracksSnapped[ix][1])
            midi_tracks.append((onset, offset, midi))

        return midi_tracks, analysisResults["beats"], analysisResults["grid"]

    def get_predominant_pitch(self):
        if self.audio != []:
            PitchMelodia = es.PitchMelodia(guessUnvoiced=self.guessUnvoiced,
                                           frameSize=self.frameSize,
                                           hopSize=self.hopSize,
                                           maxFrequency=self.maxFrequency,
                                           minFrequency=self.minFrequency,
                                           sampleRate=self.sampleRate)
            estPitch, pitchConfidence = PitchMelodia(self.audio)
            return np.array(estPitch), pitchConfidence
        else:
            return None, None

    def pitch2midi(self, pitchArray, quantizePitch=True):
        midi = []
        for pitch in pitchArray:
            if pitch == 0:
                midi.append(0)
            else:
                if quantizePitch:
                    midi.append(np.int(np.round(69 + 12 * np.math.log(pitch / 440.0, 2), 0)))
                else:
                    midi.append(69 + 12 * np.math.log(pitch / 440.0, 2))
        return midi

    def get_onsets(self):

        W = es.Windowing(type=self.winType)
        c2p = es.CartesianToPolar()
        fft = es.FFT()
        onsetDetection = es.OnsetDetection(method=self.onsetMethod, sampleRate=44100)
        onsets = es.Onsets()
        # onsetIndex = []
        pool = Pool()

        for frame in es.FrameGenerator(self.audio, frameSize=1024, hopSize=512):
            mag, phase, = c2p(fft(W(frame)))
            onsetDetection.configure(method=self.onsetMethod)
            onsetFunction = onsetDetection(mag, phase)
            pool.add("onsetFunction", onsetFunction)

        DetectedOnsetsArray = onsets([pool["onsetFunction"]], [1])

        return DetectedOnsetsArray

    def consecutive(self, data, stepsize=0):
        # source:
        # https://stackoverflow.com/questions/
        # 7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def splitUnquantizedMIDI(self, midiArrayQuantized, midiArrayUnquantized, minFrame=1):
        # minFrame specifies the minimum frame duration of a midi note to be accepted as a segment

        MIDIgroups = self.consecutive(midiArrayQuantized, stepsize=0)
        TimeStamps = []  # timeStamps in s

        groupCount = 0
        indexCount = 0
        MIDIgroupsProcessed = []
        indexStamps = []

        for groupCount in range(len(MIDIgroups)):
            MIDIgroup = MIDIgroups[groupCount]
            if min(MIDIgroup) > 0.0 and len(MIDIgroup) >= minFrame:
                indexStamp = np.arange(indexCount, indexCount + len(MIDIgroup))
                TimeStamps.append(np.arange(indexCount, indexCount + len(MIDIgroup)) *
                                  self.hopSize / float(self.sampleRate))
                indexStamps.append(indexStamp)

                MIDIgroupsProcessed.append(midiArrayUnquantized[indexStamp])

            indexCount = indexCount + len(MIDIgroup)

        return indexStamps, TimeStamps, MIDIgroupsProcessed

    def splitQuantizedMIDI(self, midiArrayQuantized, minFrame=1):
        # minFrame specifies the minimum frame duration of a midi note to be accepted as a segment
        MIDIgroups = self.consecutive(midiArrayQuantized, stepsize=0)
        TimeStamps = []  # timeStamps in s

        groupCount = 0
        indexCount = 0
        MIDIgroupsProcessed = []
        indexStamps = []
        for groupCount in range(len(MIDIgroups)):
            MIDIgroup = MIDIgroups[groupCount]
            if min(MIDIgroup) > 0.0 and len(MIDIgroup) >= minFrame:
                MIDIgroupsProcessed.append([MIDIgroup[0], MIDIgroup[0]])
                times = np.arange(indexCount, indexCount + len(MIDIgroup)) * self.hopSize / float(self.sampleRate)
                indices = np.arange(indexCount, indexCount + len(MIDIgroup))

                TimeStamps.append([times[0], times[-1]])
                indexStamps.append([indices[0], indices[-1]])
            indexCount = indexCount + len(MIDIgroup)

        return indexStamps, TimeStamps, MIDIgroupsProcessed

    def getBeats(self):
        BeatTracker = es.BeatTrackerMultiFeature()
        beats, _ = BeatTracker(self.audio)
        return beats

    def find_nearest(self, array, value):
        idx = (np.abs(np.array(array) - value)).argmin()
        return array[idx]

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

        grid = [0]

        if beatDivision == 1:
            grid = beats
        else:
            for k in range(1, beatDivision + 1):  # create the possibly missing grids before beat 0
                if beats[0] - k * gridRes > 0:
                    grid.append(beats[0] - k * gridRes)

            for i in range(len(beats)):
                if not i == len(beats) - 1:
                    gridRes = (beats[i + 1] - beats[i]) / float(beatDivision)  # resolution of grid in ms
                for k in range(beatDivision):
                    grid.append(beats[i] + k * gridRes)

        return grid

    def RemoveTracksAwayFromOnsets(self, MIDItimeStamps, MIDIGroups, onsets, minLength=.1):
        # Removes MIDI tracks that start too far away from the existing onsets
        # far away defined as more than minLength (seconds)
        # also if the majority of a track is before the nearest onset, the track is discarded

        MIDITracks = []
        MIDITrackTimes = []

        for i in range(len(MIDItimeStamps)):
            MIDItimeStamp = MIDItimeStamps[i]
            MIDIGroup = MIDIGroups[i]
            # # print (MIDItimeStamp[0])
            nearestOnset = self.find_nearest(onsets, MIDItimeStamp[0])
            checkDistance = abs(nearestOnset - MIDItimeStamp[0]) < minLength

            if (abs(MIDItimeStamp[1] - nearestOnset) - abs(nearestOnset - MIDItimeStamp[0]) > .8 * (
                    MIDItimeStamp[1] - MIDItimeStamp[0])):
                checkMajorityPostOnset = True
            else:
                checkMajorityPostOnset = False

            if checkDistance and checkMajorityPostOnset:
                MIDITracks.append(MIDIGroup)
                MIDITrackTimes.append(MIDItimeStamp)

        return MIDITrackTimes, MIDITracks

    def RemoveShortTracks(self, MIDItimeStamps, MIDIGroups, onsets, minLength=4):
        # minLength in terms of frames

        MIDITracks = []
        MIDITrackTimes = []

        for i in range(len(MIDItimeStamps)):
            MIDItimeStamp = MIDItimeStamps[i]
            MIDIGroup = MIDIGroups[i]
            if (MIDItimeStamp[1] - MIDItimeStamp[0]) * self.sampleRate > minLength * self.hopSize:
                MIDITracks.append(MIDIGroup)
                MIDITrackTimes.append(MIDItimeStamp)
        return MIDITrackTimes, MIDITracks

    def SnapToGrid(self, MIDItimeStamps, MIDIGroups, grid, ):
        MIDItimeStampsCopy = deepcopy(MIDItimeStamps)

        for i in range(len(MIDItimeStamps)):
            if self.snapBeginning:
                MIDItimeStampsCopy[i][0] = self.find_nearest(grid, MIDItimeStampsCopy[i][0])
            if self.snapEnd:
                MIDItimeStampsCopy[i][1] = self.find_nearest(grid, MIDItimeStampsCopy[i][1])

        return MIDItimeStampsCopy, MIDIGroups

    def existingMIDIValues(self, MIDIGroups):
        # returns just the midi values extracted (sorted with no duplicates)
        existingMIDIs = np.array([])

        for MIDIGroup in MIDIGroups:
            existingMIDIs = np.append(existingMIDIs, MIDIGroup[0])

        existingMIDIs = np.unique(existingMIDIs).tolist()

        return sorted(existingMIDIs)
