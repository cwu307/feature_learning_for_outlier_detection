'''
This script is to extract train a basic auto-encoder for feature learning project
Note:
    1) Sequentially train the AE (file per file)
    2) Convolutional AE
CW @ GTCMT 2017
'''

import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, UpSampling2D, BatchNormalization
from keras.models import Model

def createModel(input_dim, embedding_dim, selected_optimizer, selected_loss):
    print('using magnitude spectrogram model')
    input = Input(shape=(1, input_dim, 2304)) # 1ch, 256 x 2304
    x = Convolution2D(32, (256, 5), activation='relu', padding='same', data_format='channels_first')(input) #32 x 256 x 2304
    x = MaxPooling2D((4, 16), padding='same', data_format='channels_first')(x) #32 x 64 x 144
    x = Convolution2D(16, (64, 5), activation='relu', padding='same', data_format='channels_first')(x) #16 x 64 x 144
    x = MaxPooling2D((16, 16), padding='same', data_format='channels_first')(x) #16 x 4 x 9

    encoded = Convolution2D(embedding_dim, (1, 1), activation='relu', padding='same', data_format='channels_first')(x) #8 x 4 x 9 = 288

    out = Conv2DTranspose(4, (16, 16), padding='valid', strides=(16, 16), data_format='channels_first')(encoded) # 4 x 64 x 144
    #print(K.int_shape(out))
    out = Convolution2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(out) #32 x 64 x 144
    out = Conv2DTranspose(32, (4, 16), padding='valid', strides=(4, 16), data_format='channels_first')(out) #32 x 256  x 2304
    output = Convolution2D(1, (1, 1), activation='relu', padding='same', data_format='channels_first')(out) # 1 x 256 x 2304

    #==== create model
    autoencoder = Model(input, output)
    encoder = Model(input, encoded)

    #==== compile model
    autoencoder.compile(optimizer=selected_optimizer, loss=selected_loss, metrics=['mae'])
    return autoencoder, encoder



def createModel_cqt(input_dim, embedding_dim, selected_optimizer, selected_loss):
    print('using CQT model')
    input = Input(shape=(1, input_dim, 1024)) # 1ch, 80 x 1024
    x = Convolution2D(32, (40, 10), activation='relu', padding='same', data_format='channels_first')(input) #32 x 80 x 1024 (original 11 by 11)
    x = MaxPooling2D((2, 8), padding='same', data_format='channels_first')(x) #32 x 40 x 128
    x = Convolution2D(16, (20, 5), activation='relu', padding='same', data_format='channels_first')(x) #16 x 40 x 128 (original 5 by 5)
    x = MaxPooling2D((10, 16), padding='same', data_format='channels_first')(x) #16 x 4 x 8

    encoded = Convolution2D(embedding_dim, (1, 1), activation='relu', padding='same', data_format='channels_first')(x) #8 x 4 x 8 = 256

    out = Conv2DTranspose(embedding_dim, (10, 16), padding='valid', strides=(10, 16), data_format='channels_first')(encoded) # 8 x 40 x 128
    out = Convolution2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(out) #32 x 40 x 128
    out = Conv2DTranspose(32, (2, 8), padding='valid', strides=(2, 8), data_format='channels_first')(out) #32 x 80 x 1024
    output = Convolution2D(1, (1, 1), activation='relu', padding='same', data_format='channels_first')(out) # 1 x 80 x 1024

    #==== create model
    autoencoder = Model(input, output)
    encoder = Model(input, encoded)

    #==== compile model
    autoencoder.compile(optimizer=selected_optimizer, loss=selected_loss, metrics=['mae'])
    return autoencoder, encoder

def createModel_cqt_sigmoid(input_dim, embedding_dim, selected_optimizer, selected_loss):
    print('using CQT model with sigmoid output')
    input = Input(shape=(1, input_dim, 1024)) # 1ch, 80 x 1024
    x = Convolution2D(32, (11, 11), activation='relu', padding='same', data_format='channels_first')(input) #32 x 80 x 1024
    x = MaxPooling2D((2, 8), padding='same', data_format='channels_first')(x) #32 x 40 x 128
    x = Convolution2D(16, (5, 5), activation='relu', padding='same', data_format='channels_first')(x) #16 x 40 x 128
    x = MaxPooling2D((10, 16), padding='same', data_format='channels_first')(x) #16 x 4 x 8

    encoded = Convolution2D(embedding_dim, (1, 1), activation='relu', padding='same', data_format='channels_first')(x) #8 x 4 x 8 = 256

    out = Conv2DTranspose(embedding_dim, (10, 16), padding='valid', strides=(10, 16), data_format='channels_first')(encoded) # 8 x 40 x 128
    out = Convolution2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(out) #32 x 40 x 128
    out = Conv2DTranspose(32, (2, 8), padding='valid', strides=(2, 8), data_format='channels_first')(out) #32 x 80 x 1024
    output = Convolution2D(1, (1, 1), activation='sigmoid', padding='same', data_format='channels_first')(out) # 1 x 80 x 1024

    #==== create model
    autoencoder = Model(input, output)
    encoder = Model(input, encoded)

    #==== compile model
    autoencoder.compile(optimizer=selected_optimizer, loss=selected_loss, metrics=['mae'])
    return autoencoder, encoder


'''
symmetric means the input representation has been reshaped to a symmetric squared matrix
the entire CQT matrix has been sub-divided into smaller matrices
'''
def createModel_cqt_symmetric(input_dim, embedding_dim, selected_optimizer, selected_loss):
    input = Input(shape=(1, input_dim, 80)) # 1ch, 80 x 80
    x = Convolution2D(64, (11, 11), strides= (2, 2), activation='relu', padding='same', data_format='channels_first')(input) #64 x 40 x 40
    x = Convolution2D(32, (5, 5), strides= (2, 2), activation='relu', padding='same', data_format='channels_first')(x) #32 x 20 x 20
    x = Convolution2D(16, (3, 3), strides= (2, 2), activation='relu', padding='same', data_format='channels_first')(x) #16 x 10 x 10
    x = Convolution2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(x) #8 x 5 x  5

    encoded = Convolution2D(embedding_dim, (1, 1), strides= (1, 1), activation='relu', padding='same', data_format='channels_first')(x) #2 x 5 x 5

    out = Convolution2D(8, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(encoded) #8 x 5 x 5
    out = UpSampling2D((2, 2), data_format='channels_first')(out)  # 8 x 10 x 10
    out = Convolution2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(out)  # 16 x 10 x 10
    out = UpSampling2D((2, 2), data_format='channels_first')(out)  # 32 x 20 x 20
    out = Convolution2D(32, (5, 5), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(out)  # 32 x 20 x 20
    out = UpSampling2D((2, 2), data_format='channels_first')(out)  # 16 x 40 x 40
    out = Convolution2D(64, (11, 11), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(out)  # 64 x 40 x 40
    out = UpSampling2D((2, 2), data_format='channels_first')(out)  # 64 x 80 x 80
    output = Convolution2D(1, (1, 1), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(out)

    print(np.shape(output))
    #==== create model
    autoencoder = Model(input, output)
    encoder = Model(input, encoded)

    #==== compile model
    autoencoder.compile(optimizer=selected_optimizer, loss=selected_loss, metrics=['mae'])
    return autoencoder, encoder

'''
Testing similar architecture as described in Keunwoochoi's paper
'''
def createModel_cqt_ae(input_dim, embedding_dim, selected_optimizer, selected_loss):
    print('using magnitude CQT model')
    input = Input(shape=(1, input_dim, 1290)) #1 x 80 x 1290
    out1 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(input) #32 x 80 x 1290
    out1 = BatchNormalization()(out1)
    out1 = MaxPooling2D((2, 5), padding='same', data_format='channels_first')(out1)  #32 x 40 x 258
    out2 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(out1) #32 x 40 x 258
    out2 = BatchNormalization()(out2)
    out2 = MaxPooling2D((2, 3), padding='same', data_format='channels_first')(out2) #32 x 20 x 86
    out3 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(out2) #32 x 20 x 86
    out3 = BatchNormalization()(out3)
    out3 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out3)  #32 x 10 x 43

    encoded = Convolution2D(embedding_dim, (1, 1), activation='relu', padding='same', data_format='channels_first')(out3) #embed_dim x 10 x 43
    encoded = BatchNormalization()(encoded)

    out4 = UpSampling2D((2, 2), data_format='channels_first')(encoded)  #32 x 20 x 86
    out4 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(out4)  #32 x 20 x 86
    out4 = BatchNormalization()(out4)
    out5 = UpSampling2D((2, 3), data_format='channels_first')(out4)  #32 x 40 x 258
    out5 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(out5)  #32 x 40 x 258
    out5 = BatchNormalization()(out5)
    out6 = UpSampling2D((2, 5), data_format='channels_first')(out5)  #32 x 80 x 1290
    out6 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(out6)  # 32 x 40 x 258
    out6 = BatchNormalization()(out6)

    output = Convolution2D(1, (1, 1), activation='relu', padding='same', data_format='channels_first')(out6) #1 x 80 x 1290



    #==== create model
    autoencoder = Model(input, output)
    layer1_extractor = Model(input, out1)
    layer2_extractor = Model(input, out2)
    layer3_extractor = Model(input, out3)
    bottleneck_extractor = Model(input, encoded)


    #==== compile model
    autoencoder.compile(optimizer=selected_optimizer, loss=selected_loss, metrics=['mae'])
    return autoencoder, layer1_extractor, layer2_extractor, layer3_extractor, bottleneck_extractor