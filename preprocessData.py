'''
This script is to preprocess the data for training deep auto-encoder
CW @ GTCMT 2017
'''

import numpy as np
from FileUtil import getFilePathList
from scipy.io import loadmat
from os import listdir
from librosa.core import cqt
from librosa.core import load

REALMIN = np.finfo(np.double).tiny

def getConcatenateSpectrogram(target_path):
    #==== loop through all data, concatenate all spectrograms, and save as .npy
    genre_folders = listdir(target_path)
    X_train = []
    print('start looping...\n')
    c = 0
    for single_genre_folder in genre_folders:
        print('Processing genre = %s' % single_genre_folder)
        if single_genre_folder[0] is not '.':
            file_list = getFilePathList(target_path + single_genre_folder + '/', 'mat')
            for file in file_list:
                tmp = loadmat(file) # the variable is named Y in matlab
                X = tmp['Y'] # X is the complex spectrogram from STFT
                X = X[0:256, 0:2490]
                X = np.expand_dims(X, axis=0)
                if len(X_train) == 0:
                    X_train = abs(X)
                else:
                    c += 1
                    print(c)
                    X_train = np.concatenate((X_train, abs(X)), axis=0)
    print(np.shape(X_train))
    return X_train

def getConcatenateCQT(target_path):
    # ==== loop through all data, concatenate all CQT, and save as .npy
    genre_folders = listdir(target_path)
    X_train = []
    print('start looping...\n')
    c = 0
    for single_genre_folder in genre_folders:
        print('Processing genre = %s' % single_genre_folder)
        if single_genre_folder[0] is not '.':
            file_list = getFilePathList(target_path + single_genre_folder + '/', 'mat')
            for file in file_list:
                tmp = loadmat(file) # the variable is named Y in matlab
                y = tmp['y']
                #print('normalize waveform by its absolute maximum value\n')
                y = np.divide(y, np.max(abs(y)))
                fs = tmp['fs']
                assert(fs == 22050)
                y = np.asarray(np.reshape(y, (len(y),)))
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


#==== define path to the dataset
target_path = '../../../data/metaData/gtzan_wav/'
save_path_data = '../../../data/metaData/gtzan_cqt_maxnorm.npy'
X_train = getConcatenateCQT(target_path)
print('now saving the data...')
np.save(save_path_data, X_train)




