'''
This script is to extract train a basic CNN for feature learning project (unsupervised)
CW @ GTCMT 2017
'''

import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dnnModels import createModel_cqt_ae
from FileUtil import getFilePathList, standardizeTensorTrackwise, normalizeTensorTrackwiseL1
preprocessingFlag = False
pdfFlag = True

#==== define data path
data_folder = '../../../data/metaData/fma_medium_cqt_recombined/'
metadata_path = '../../../data/fma_metadata/fma_medium_metadata_cw_parsed.npy'
check_path = './trained_models_unsupervised/checkpoint.h5'
ae_path = './trained_models_unsupervised/ae.h5'
ext1_path = './trained_models_unsupervised/ext1.h5'
ext2_path = './trained_models_unsupervised/ext2.h5'
ext3_path = './trained_models_unsupervised/ext3.h5'
ext4_path = './trained_models_unsupervised/ext4.h5'
ext5_path = './trained_models_unsupervised/ext5.h5'

#==== define DNN parameters
input_dim = 80
input_dim2 = 1280
embed_dim = 32
num_epochs = 60
selected_optimizer = Adam(lr=0.0001)
selected_loss = 'mean_squared_logarithmic_error'
checker = ModelCheckpoint(check_path)
tbcallback = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=False)
earlyStop  = EarlyStopping(monitor='loss', patience=5, mode='min')
reduce_lr  = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00000001)
ae, ext1, ext2, ext3, ext4, ext5 = createModel_cqt_ae(input_dim, input_dim2, embed_dim, selected_optimizer, selected_loss)

all_list = getFilePathList(data_folder, '.npy')
all_list = sorted(all_list)

all_metadata = np.load(metadata_path)
genre_num_medium = all_metadata[1]
y_train = genre_num_medium
y_train = to_categorical(y_train)

for e in range(0, num_epochs):
    print("epoch %d" % e)
    p = 0
    for data_path in all_list:
        print(data_path)
        X_train = np.load(data_path) #5000 x 80 x 1280
        istart = p * 5000
        iend   = istart + 5000
        y_train_sub = y_train[istart:iend]
        print(istart)
        print(iend)
        p += 1
        if preprocessingFlag:
            print('Warning: data preprocessing is on')
            X_train = 10 * np.log10(np.maximum(X_train, 10e-6))
            X_train =  X_train - np.max(X_train)
            X_train = np.maximum(X_train, -80)
            X_train = standardizeTensorTrackwise(X_train)
        if pdfFlag:
            print('Normalize T-F representation as joint pdf')
            X_train = normalizeTensorTrackwiseL1(X_train)
        X_train = np.expand_dims(X_train, axis=1) #5000 x 1 x 80 x 1280
        ae.fit(X_train, X_train, epochs=1, batch_size=4, callbacks=[checker, tbcallback, earlyStop], verbose=1, shuffle=True)

#==== save results
ae.save(ae_path)
ext1.save(ext1_path)
ext2.save(ext2_path)
ext3.save(ext3_path)
ext4.save(ext4_path)
ext5.save(ext5_path)