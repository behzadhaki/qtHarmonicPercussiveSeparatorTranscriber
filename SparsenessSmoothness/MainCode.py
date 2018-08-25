from DecomposeSmoothSparse import HPSS           # Import SmoothSparse NMF separator (developed by me)
import essentia.standard as es

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

        hpss = HPSS(
            x,
            directory=saveFolderLoc,                            # Directory to save separated files
            filename=filename+"_smsp",                          # Filename used as a prefix to save the separated parts
            format="mp3",                                       # Format to save the separated files
            beta=1.5,                                           # beta divergence coefficient
            frameSize=frameSize,
            hopSize=hopSize,
            fftSize=fftSize,
            Rp=150,                                             # Number of percussive bases
            Rh=150,                                             # Number of harmonic bases
            maxIter=200.0,                                      # Number of iterations to update the har/per masks
            K_SSM=1,                                           # Percussive Spectral Smoothness weight on cost function
            K_TSP=1,                                           # Percussive Temporal Smoothness weight on cost function
            K_SSP=1,                                           # Harmonic Spectral Smoothness weight on cost function
            K_TSM=1,                                           # Harmonic Temporal Smoothness weight on cost function

        )

        hpss.separate()
        hpss.save_separated_audiofiles()
