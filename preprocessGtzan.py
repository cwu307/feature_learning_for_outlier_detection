'''
This script is to preprocess the data for training deep auto-encoder
CW @ GTCMT 2017
'''

import numpy as np
from FileUtil import getFilePathList
from scipy.io import loadmat
from os import listdir
from librosa.core import cqt, stft, load
from librosa.feature import melspectrogram
GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def getConcatenateCQT(target_path):
    # ==== loop through all data, concatenate all log magnitude CQT, and save as .npy
    if 'gtzan' in target_path:
        genre_folders = GTZAN_GENRES
    else:
        genre_folders = listdir(target_path)
        genre_folders = sorted(genre_folders)
    
    X_train = []
    print('start looping...\n')
    c = 0
    for single_genre_folder in genre_folders:
        print('Processing genre = %s' % single_genre_folder)
        if single_genre_folder[0] is not '.':
            file_list = getFilePathList(target_path + single_genre_folder + '/', 'au')
            for filepath in file_list:
                y, sr = load(filepath, sr=22050, mono=True)
                #print('normalize waveform by its absolute maximum value\n')
                y = np.divide(y, np.max(abs(y)))
                fs = sr
                assert(fs == 22050)
                X_cqt = cqt(y, sr=22050, hop_length=512, n_bins=80, bins_per_octave=12)
                X_cqt = X_cqt[:, 0:1290]
                X_cqt = np.expand_dims(X_cqt, axis=0)
                #print(np.shape(X_cqt))
                if len(X_train) == 0:
                    X_train = abs(X_cqt)
                else:
                    c += 1
                    print(c)
                    X_train = np.concatenate((X_train, abs(X_cqt)), axis=0)
    print(np.shape(X_train))
    return X_train

def getConcatenateMelspectro(target_path):
    if 'gtzan' in target_path:
        genre_folders = GTZAN_GENRES
    else:
        genre_folders = listdir(target_path)
        genre_folders = sorted(genre_folders)
    X_train = []
    print('start looping...\n')
    c = 0
    for single_genre_folder in genre_folders:
        print('Processing genre = %s' % single_genre_folder)
        if single_genre_folder[0] is not '.':
            file_list = getFilePathList(target_path + single_genre_folder + '/', 'au')
            for filepath in file_list:
                y, sr = load(filepath, sr=22050, mono=True)
                #print('normalize waveform by its absolute maximum value\n')
                y = np.divide(y, np.max(abs(y)))
                fs = sr
                assert(fs == 22050)
                X_mel = melspectrogram(y, sr=sr, n_fft=1024, hop_length=512, n_mels=96, fmin=0.0, fmax=10000, power=2.0)
                print(np.shape(X_mel))
                X_mel = X_mel[:, 0:1290]
                X_mel = np.expand_dims(X_mel, axis=0)
                if len(X_train) == 0:
                    X_train = abs(X_mel)
                else:
                    c += 1
                    print(c)
                    X_train = np.concatenate((X_train, abs(X_mel)), axis=0)
    print(np.shape(X_train))
    return X_train

def main():
    #==== define path to the dataset
    target_path = '/data/gtzan/'
    X_train = getConcatenateMelspectro(target_path)
    np.save('/dataz/preprop_data/gtzan_tf/gtzan_mel96_maxnorm.npy', X_train)
    print('finished')
    return ()

if __name__ == "__main__":
    print('running main() directly')
    main()




