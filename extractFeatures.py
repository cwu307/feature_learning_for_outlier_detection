'''
This script is to test the trained encoder
CW @ GTCMT 2017
'''

import numpy as np
from keras.models import load_model
from FileUtil import averageActivationMap


#==== check the directories
data_path = '../../../data/metaData/gtzan_cqt_maxnorm_log.npy'
ae_path = './trained_models/ae.h5'
ext1_path = './trained_models/ext1.h5'
ext2_path = './trained_models/ext2.h5'
ext3_path = './trained_models/ext3.h5'
ext4_path = './trained_models/ext4.h5'
save_path = '../metaData/gtzan_features_1000by_128.npy'


#==== check the dimensionality
X_train = np.load(data_path)


X_train = np.expand_dims(X_train, axis=1)

ext1_model = load_model(ext1_path)
ext2_model = load_model(ext2_path)
ext3_model = load_model(ext3_path)
ext4_model = load_model(ext4_path)

#32 x 4 = 128
lay1 = ext1_model.predict(X_train, batch_size=1) #1000 x 32 x m1 x m2
lay2 = ext2_model.predict(X_train, batch_size=1)
lay3 = ext3_model.predict(X_train, batch_size=1)
lay4 = ext4_model.predict(X_train, batch_size=1)

m1 = averageActivationMap(lay1)
m2 = averageActivationMap(lay2)
m3 = averageActivationMap(lay3)
m4 = averageActivationMap(lay4)

X = np.concatenate((m1, m2, m3, m4), axis=1)

np.save(save_path, X)
