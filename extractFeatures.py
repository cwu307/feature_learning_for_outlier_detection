'''
This script is to test the trained encoder
CW @ GTCMT 2017
'''

import numpy as np
from keras.models import load_model
from FileUtil import averageActivationMap, standardizeTensorTrackwise, normalizeTensorTrackwiseL1, convert2dB
preprocessingFlag = True
pdfFlag = False

#==== check the directories
data_path = '../../../data/metaData/gtzan_cqt80_maxnorm.npy'
ext1_path = './trained_models/ext1.h5'
ext2_path = './trained_models/ext2.h5'
ext3_path = './trained_models/ext3.h5'
ext4_path = './trained_models/ext4.h5'
ext5_path = './trained_models/ext5.h5'
# ext1_path = './trained_models_unsupervised/ext1.h5'
# ext2_path = './trained_models_unsupervised/ext2.h5'
# ext3_path = './trained_models_unsupervised/ext3.h5'
# ext4_path = './trained_models_unsupervised/ext4.h5'
# ext5_path = './trained_models_unsupervised/ext5.h5'
save_path = '../../../data/metaData/gtzan_features_1000by160_learned_from_fma_medium_30ep_mae.npy'


#==== check the dimensionality
X_train = np.load(data_path) #1000 x 80 x 1290
X_train = X_train[:,:,0:1280]
if preprocessingFlag:
    print('Warning: data preprocessing is on')
    # X_train = 10 * np.log10(np.maximum(X_train, 10e-6))
    # X_train = X_train - np.max(X_train)
    # X_train = np.maximum(X_train, -80)
    X_train = convert2dB(X_train)
    X_train = standardizeTensorTrackwise(X_train)
if pdfFlag:
    print('Normalize T-F representation as joint pdf')
    X_train = normalizeTensorTrackwiseL1(X_train)
X_train = np.expand_dims(X_train, axis=1)

ext1_model = load_model(ext1_path)
ext2_model = load_model(ext2_path)
ext3_model = load_model(ext3_path)
ext4_model = load_model(ext4_path)
ext5_model = load_model(ext5_path)

#32 x 4 = 128
lay1 = ext1_model.predict(X_train, batch_size=1) #1000 x 32 x m1 x m2
lay2 = ext2_model.predict(X_train, batch_size=1)
lay3 = ext3_model.predict(X_train, batch_size=1)
lay4 = ext4_model.predict(X_train, batch_size=1)
lay5 = ext5_model.predict(X_train, batch_size=1)

m1 = averageActivationMap(lay1)
m2 = averageActivationMap(lay2)
m3 = averageActivationMap(lay3)
m4 = averageActivationMap(lay4)
m5 = averageActivationMap(lay5)
X = np.concatenate((m1, m2, m3, m4, m5), axis=1)

print(np.shape(X))
print('saving file to  %s' % save_path)
np.save(save_path, X)
