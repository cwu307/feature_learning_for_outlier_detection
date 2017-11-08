'''
This script is to extract train a basic auto-encoder for feature learning project
CW @ GTCMT 2017
'''

import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dnnModels import createModel_cqt_classification_fma_medium
from FileUtil import getFilePathList, standardizeTensorTrackwise
preprocessingFlag = False

#==== define data path
data_folder = '../../../data/metaData/fma_cqt/'
metadata_path = '../../../data/fma_metadata/fma_medium_metadata_cw_parsed.npy'
check_path = './trained_models/checkpoint.h5'
ae_path = './trained_models/ae.h5'
ext1_path = './trained_models/ext1.h5'
ext2_path = './trained_models/ext2.h5'
ext3_path = './trained_models/ext3.h5'
ext4_path = './trained_models/ext4.h5'

#==== define DNN parameters
input_dim = 80
input_dim2 = 1280
num_epochs = 60
selected_optimizer = Adam(lr=0.0001)
selected_loss = 'accuracy'
checker = ModelCheckpoint(check_path)
tbcallback = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=True)
earlyStop  = EarlyStopping(monitor='loss', patience=3, mode='min')
reduce_lr  = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00000001)
classifier, ext1, ext2, ext3, ext4 = createModel_cqt_classification_fma_medium(input_dim, input_dim2, selected_optimizer, selected_loss)

all_list = getFilePathList(data_folder, '.npy')
all_list = sorted(all_list)

all_metadata = np.load(metadata_path)
genre_num_medium = all_metadata[1]
y_train = genre_num_medium
y_train = to_categorical(y_train)

p = 0
for data_path in all_list:

    print('now training on:')
    print(data_path)
    X_train = np.load(data_path) #1000 x 80 x 1290
    istart = p * 1000
    iend   = istart + 1000
    y_train_sub = y_train[istart:iend]
    p += 1
    if preprocessingFlag:
        print('Warning: data preprocessing is on')
        X_train = np.log10(X_train + 10e-10)
        X_train = standardizeTensorTrackwise(X_train)
    X_train = np.expand_dims(X_train, axis=1) #1000 x 1 x 80 x 1290
    classifier.fit(X_train, y_train_sub, epochs=num_epochs, batch_size=4, callbacks=[checker, tbcallback])

#==== save results
ae.save(ae_path)
ext1.save(ext1_path)
ext2.save(ext2_path)
ext3.save(ext3_path)
ext4.save(ext4_path)
