'''
This script is to extract train a basic auto-encoder for feature learning project
CW @ GTCMT 2017
'''

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dnnModels import createModel_log_cqt
from FileUtil import scaleMatrix, reshapeInputTensor

#==== define data path
data_path = '../../../data/metaData/gtzan_cqt_maxnorm.npy'
check_path = './trained_models/checkpoint.h5'
ae_path = './trained_models/ae.h5'
ext1_path = './trained_models/ext1.h5'
ext2_path = './trained_models/ext2.h5'
ext3_path = './trained_models/ext3.h5'
ext4_path = './trained_models/ext4.h5'

#==== define DNN parameters
X_train = np.load(data_path) #1000 x 80 x 1290
input_dim = 80
embedding_dim = 32
num_epochs = 1000
selected_optimizer = Adam(lr=0.0001)
selected_loss = 'mse'
checker = ModelCheckpoint(check_path)
tbcallback = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=True)
earlyStop  = EarlyStopping(monitor='loss', patience=3, mode='min')
reduce_lr  = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00001)
X_train = np.expand_dims(X_train, axis=1) #1000 x 1 x 80 x 1290

ae, ext1, ext2, ext3, ext4 = createModel_log_cqt(input_dim, embedding_dim, selected_optimizer, selected_loss)
ae.fit(X_train, X_train, epochs=num_epochs, batch_size=4, callbacks=[checker, tbcallback, reduce_lr, earlyStop])

#==== save results
ae.save(ae_path)
ext1.save(ext1_path)
ext2.save(ext2_path)
ext3.save(ext3_path)
ext4.save(ext4_path)
