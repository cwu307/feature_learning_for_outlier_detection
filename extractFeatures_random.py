'''
This script is to test the trained encoder
CW @ GTCMT 2017
'''

import numpy as np
from keras.optimizers import Adam
from FileUtil import averageActivationMap, standardizeTensorTrackwise, standardizeEntireTensor, convert2dB, scaleTensorTrackwise
from dnnModels import createModel_cqt_random
preprocessingFlag = True

#==== check the directories
#data_path = '../../../data/metaData/gtzan_cqt80_maxnorm.npy'
data_path = '../../../data/metaData/reorganized_features/gtzan_mel96_maxnorm.npy'
save_path = '../../../data/metaData/gtzan_features_1000by160_elu_mel96_random.npy'

#==== check the dimensionality
X_train = np.load(data_path) #1000 x 80 x 1290
X_train = X_train[:,:,0:1280]
if preprocessingFlag:
    print('Warning: data preprocessing is on')
    X_train = convert2dB(X_train)
    X_train = standardizeTensorTrackwise(X_train)
X_train = np.expand_dims(X_train, axis=1)

input_dim = 96
input_dim2 = 1280
selected_optimizer = Adam(lr=0.0001)
selected_loss = 'categorical_crossentropy'
classifier, ext1, ext2, ext3, ext4, ext5 = createModel_cqt_random(input_dim, input_dim2, selected_optimizer, selected_loss)

lay1 = ext1.predict(X_train, batch_size=1) #1000 x 32 x m1 x m2
lay2 = ext2.predict(X_train, batch_size=1)
lay3 = ext3.predict(X_train, batch_size=1)
lay4 = ext4.predict(X_train, batch_size=1)
lay5 = ext5.predict(X_train, batch_size=1)

m1 = averageActivationMap(lay1)
m2 = averageActivationMap(lay2)
m3 = averageActivationMap(lay3)
m4 = averageActivationMap(lay4)
m5 = averageActivationMap(lay5)
#m5 = lay5

X = np.concatenate((m1, m2, m3, m4, m5), axis=1)


print(np.shape(X))
print('saving file to  %s' % save_path)
np.save(save_path, X)
