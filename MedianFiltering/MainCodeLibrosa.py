from librosa.core import stft, istft                    # Librosa Version 0.6.0
from librosa.decompose import hpss
import essentia.standard as es
from essentia import array

import os, glob

dataSetLocation = "../audio/"                  # Location of dataset

nFiles = 2                                 # Number of entries in dataset

mixFolder = "/mix/"                          # folder containing the file to be separated

saveFolder = "/separated/"                   # Subdirectory in each entry to save the results

nSeconds = 30                               # number of seconds to analyze


frameSize = 2048                                           # frame size for stft calculation
hopSize = 1024                                             # hop size for stft calculation
fftSize = 2048                                             # fft size for stft calculation
winType = "hann"                                           # window type for stft


for i in range(nFiles):
    print("Separating harmonic and percussive parts in entry", str(i+1), "of dataset")

    mp3mixFolder = dataSetLocation+str(i+1)+mixFolder

    mixFiles = glob.glob(mp3mixFolder+"*.mp3")         # filename of the percussive and harmonic mixture

    for mixFile in mixFiles:
        filename = mixFile.split("/")[-1][:-4]

        saveFolderLoc = dataSetLocation+str(i+1)+saveFolder  # filename of the percussive and harmonic mixture

        if not os.path.isdir(saveFolderLoc):
            os.makedirs(saveFolderLoc)

        monoLoader = es.MonoLoader(filename=mixFile, sampleRate=44100)
        x = monoLoader()[:nSeconds*44100]

        _stft = stft(x, n_fft=fftSize, hop_length=hopSize, win_length=frameSize, window=winType)

        X_H, X_P = hpss(_stft, kernel_size=150)                     # Get harmonic and percussive stfts

        x_h = istft(X_H, hop_length=hopSize, win_length=frameSize)  # Convert stfts to time domain signals
        x_p = istft(X_P, hop_length=hopSize, win_length=frameSize)

        MonoWriter = es.MonoWriter(sampleRate=44100, format="mp3")  # Write to file
        MonoWriter.configure(filename=saveFolderLoc+filename+"_median_percussive.mp3")
        MonoWriter(array(x_p))

        MonoWriter = es.MonoWriter(sampleRate=44100, format="mp3")  # Write to file
        MonoWriter.configure(filename=saveFolderLoc+filename+"_median_harmonic.mp3")
        MonoWriter(array(x_h))

print("DONE")