'''
Split FMA dataset based on the given metadata
'''
import numpy as np
import glob
from os import listdir, mkdir
from os.path import isdir
from librosa.core import cqt, stft, load
from librosa.feature import melspectrogram

def concateMatrix(X_all, X_cur):
    if len(X_all) == 0:
        X_all = X_cur
    else:
        X_all = np.concatenate((X_all, X_cur), axis=0)
    return X_all

selected_data = 'medium'
print('start processing fma -- %s subset...' % selected_data)
#============================
if selected_data == 'medium':
    fma_audio_path = '/data/fma_medium/'
    preprop_data_path = '/dataz/preprop_data/fma_medium_mel/'
    fma_metadata_path = '/data/fma_metadata/fma_medium_metadata_cw_parsed.npy'
elif selected_data == 'small':
    fma_audio_path = '/data/fma_small/'
    fma_metadata_path = '/data/fma_metadata/fma_small_metadata_cw_parsed.npy'

all_metadata = np.load(fma_metadata_path)
track_ids = all_metadata[0]
genre_num = all_metadata[1]
genre_dic = all_metadata[2]
subset_num = all_metadata[3]
subset_dic = all_metadata[4]
splits_num = all_metadata[5] #0: test, 1:training, 2:validation
splits_dic = all_metadata[6]

clean_list = sorted(glob.glob(fma_audio_path + '*/*.mp3')) 

print('creating directories for saving new splits...')
train_path = preprop_data_path + 'training/'
test_path = preprop_data_path + 'test/'
val_path = preprop_data_path + 'validation/'
if not isdir(train_path):
    mkdir(train_path)
if not isdir(test_path):
    mkdir(test_path)
if not isdir(val_path):
    mkdir(val_path)

current_track_count = 0
X_train = []
y_train = []
train_count = 0

X_test = []
y_test = []
test_count = 0

X_val = []
y_val = []
val_count = 0

for single_file in clean_list:
    y, sr = load(single_file, sr=22050, mono=True)
    if len(y)/float(sr) < 29 or np.max(abs(y)) == 0:
        print('too short or slience, skip!')
        continue
    y = np.divide(y, np.max(abs(y)))
    X_mel = melspectrogram(y, sr=sr, n_fft=1024, hop_length=512, n_mels=96, fmin=0.0, fmax=10000, power=2.0)
    numFreq, numBlock = np.shape(X_mel)
    X_mel = X_mel[:, 0:1290]
    print('currently processing track %d ...' % current_track_count)
    X_cur = np.expand_dims(X_mel, axis=0)
    y_cur = genre_num[current_track_count]
    if splits_num[current_track_count] == 0:
        y_test.append(y_cur)
        X_test = concateMatrix(X_test, X_cur)
    if splits_num[current_track_count] == 1:
        y_train.append(y_cur)
        X_train = concateMatrix(X_train, X_cur)
    if splits_num[current_track_count] == 2:
        y_val.append(y_cur)
        X_val = concateMatrix(X_val, X_cur)

    current_track_count += 1
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(X_val))

    if np.size(X_test, 0) == 3000:
        test_count += 1
        save_path = test_path + str(test_count) + '_fma_' + selected_data + '_test_X.npy'
        save_path_label = test_path + str(test_count) + '_fma_' + selected_data + '_test_y.npy'
        np.save(save_path, X_test)
        np.save(save_path_label, y_test)
        X_test = []
        y_test = []
    if np.size(X_train, 0) == 3000:
        train_count += 1
        save_path = train_path + str(train_count) + '_fma_' + selected_data + '_train_X.npy'
        save_path_label  = train_path + str(train_count) + '_fma_' + selected_data + '_train_y.npy'
        np.save(save_path, X_train)
        np.save(save_path_label, y_train)
        X_train = []
        y_train = []
    if np.size(X_val, 0) == 3000:
        val_count += 1
        save_path = val_path + str(val_count) + '_fma_' + selected_data + '_val_X.npy'
        save_path_label = val_path + str(val_count) + '_fma_' + selected_data + '_val_y.npy'
        np.save(save_path, X_val)
        np.save(save_path_label, y_val)
        X_val = []
        y_val = []
        

print('save the remaining data')
if np.size(X_test, 0) != 0:
    test_count += 1
    save_path = test_path + str(test_count) + '_fma_' + selected_data + '_test_X.npy'
    save_path_label = test_path + str(test_count) + '_fma_' + selected_data + '_test_y.npy'
    np.save(save_path, X_test)
    np.save(save_path_label, y_test)
    X_test = []
    y_test = []
if np.size(X_train, 0) != 0:
    train_count += 1
    save_path = train_path + str(train_count) + '_fma_' + selected_data + '_train_X.npy'
    save_path_label  = train_path + str(train_count) + '_fma_' + selected_data + '_train_y.npy'
    np.save(save_path, X_train)
    np.save(save_path_label, y_train)
    X_train = []
    y_train = []
if np.size(X_val, 0) != 0:
    val_count += 1
    save_path = val_path + str(val_count) + '_fma_' + selected_data + '_val_X.npy'
    save_path_label = val_path + str(val_count) + '_fma_' + selected_data + '_val_y.npy'
    np.save(save_path, X_val)
    np.save(save_path_label, y_val)
    X_val = []
    y_val = []

print('all done!')



    


