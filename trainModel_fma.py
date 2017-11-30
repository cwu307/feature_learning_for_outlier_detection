'''
This script is to extract train a basic CNN for feature learning project
CW @ GTCMT 2017
'''

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dnnModels import createModel_cqt_classification_fma_medium
from FileUtil import getFilePathList, standardizeTensorTrackwise, convert2dB
preprocessingFlag = True

#==== define data path
train_data_folder = '../../../data/metaData/fma_medium_cqt_recombined/training/'
validation_data_folder = '../../../data/metaData/fma_medium_cqt_recombined/validation/'
test_data_folder = '../../../data/metaData/fma_medium_cqt_recombined/test/'

metadata_path = '../../../data/fma_metadata/fma_small_metadata_cw_parsed.npy'
check_path = './trained_models/checkpoint.h5'
classifier_path = './trained_models/ae.h5'
ext1_path = './trained_models/ext1.h5'
ext2_path = './trained_models/ext2.h5'
ext3_path = './trained_models/ext3.h5'
ext4_path = './trained_models/ext4.h5'
ext5_path = './trained_models/ext5.h5'

#==== define DNN parameters
input_dim = 80
input_dim2 = 1280
num_epochs = 30
selected_optimizer = Adam(lr=0.0001)
selected_loss = 'categorical_crossentropy'
checker = ModelCheckpoint(check_path, monitor='val_loss', save_best_only=True)
tbcallback = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=False)
earlyStop  = EarlyStopping(monitor='val_loss', patience=2, mode='min')
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)
classifier, ext1, ext2, ext3, ext4, ext5 = createModel_cqt_classification_fma_medium(input_dim, input_dim2, selected_optimizer, selected_loss)

X_val = np.load(validation_data_folder + '1_fma_medium_val_X.npy'  )
y_val = np.load(validation_data_folder + '1_fma_medium_val_y.npy')
X_test = np.load(test_data_folder + '1_fma_medium_test_X.npy')
y_test = np.load(test_data_folder + '1_fma_medium_test_y.npy')

if preprocessingFlag:
    print('Warning: data preprocessing is on: processing validation data')
    X_val = convert2dB(X_val)
    X_val = standardizeTensorTrackwise(X_val)
    print('Warning: data preprocessing is on: processing test data')
    X_test = convert2dB(X_test)
    X_test = standardizeTensorTrackwise(X_test)

X_val = np.expand_dims(X_val, axis=1)
y_val = to_categorical(y_val, num_classes=16)
X_test = np.expand_dims(X_test, axis=1)
y_test = to_categorical(y_test, num_classes=16)

for e in range(0, num_epochs):
    print("==== epoch %d ====" % e)
    for i in range(1, 8):
        if preprocessingFlag:
            X_train_path = train_data_folder + str(i) + '_fma_medium_train_X.npy'   
            y_train_path = train_data_folder + str(i) + '_fma_medium_train_y.npy'
            print(X_train_path)
            X_train = np.load(X_train_path)
            y_train = np.load(y_train_path)
            X_train = convert2dB(X_train)
            X_train = standardizeTensorTrackwise(X_train)
            X_train = np.expand_dims(X_train, axis=1) #3000 x 1 x 80 x 1280
            y_train = to_categorical(y_train, num_classes=16)

        classifier.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=1, batch_size=2, callbacks=[checker, tbcallback, earlyStop], verbose=1, shuffle=True)
        loss, metric = classifier.evaluate(X_test, y_test, batch_size=1, verbose=1)
        print('test accuracy = %f\n' % metric)

#==== save results
classifier.save(classifier_path)
ext1.save(ext1_path)
ext2.save(ext2_path)
ext3.save(ext3_path)
ext4.save(ext4_path)
ext5.save(ext5_path)