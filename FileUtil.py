# A collection of some of the utility functions
# CW @ GTCMT 2017

from os import listdir
import numpy as np

'''
input:
    parentFolder: string, directory to the parent folder
    ext: string, extension name of the interested files
output:
    filePathList: list, directory to the files
'''
def getFilePathList(folderpath, ext):
    allfiles = listdir(folderpath)
    filePathList = []
    for file in allfiles:
        if ext in file:
            filepath = folderpath + file
            filePathList.append(filepath)
    return filePathList

'''
input:
    vector: N by 1 float vector
output:
    vector_scaled: N by 1 float vector max = 1 min = 0
    maxValue: max value in original data vector
    minValue: min value in original data vector
'''
def scaleVector(vector):
    maxValue = max(vector)
    minValue = min(vector)
    vector_scaled = np.divide(np.subtract(vector, minValue), float(np.subtract(maxValue, minValue)))
    return vector_scaled, maxValue, minValue


'''
input:
    vector: N by 1 float vector
    maxValue: max value in original data vector
    minValue: min value in original data vector
output:
    vector_scaled: N by 1 float vector max = 1 min = 0
'''
def scaleVectorWithMinMax(vector, maxValue, minValue):
    vector_scaled = np.divide(np.subtract(vector, minValue), float(np.subtract(maxValue, minValue)))
    return vector_scaled


'''
input:
    matrix: N by M float matrix, N is the #blocks, M is the #freq bins
output:
    matrix_scaled: N by M float matrix
    maxVector: 1 by M float vector of max values for each bin
    minVector: 1 by M float vector of min values for each bin
'''
def scaleMatrix(matrix):
    maxVector = np.max(matrix, axis=0)
    minVector = np.min(matrix, axis=0)
    matrix_scaled = np.divide(np.subtract(matrix, minVector), np.subtract(maxVector, minVector))
    return matrix_scaled, maxVector, minVector

'''
input:
    matrix: N by M float matrix, N is the #blocks, M is the #freq bins
    maxVector: 1 by M float vector of max values for each bin
    minVector: 1 by M float vector of min values for each bin
output:
    matrix_scaled: N by M float matrix
'''

def scaleMatrixWithMinMax(matrix, maxVector, minVector):
    matrix_scaled = np.divide(np.subtract(matrix, minVector), np.subtract(maxVector, minVector))
    return matrix_scaled


'''
input:
    data_tensor: numSample x numFreq x numBlock
output:
    data_tensor_reshaped: numSample_mod x numFreq x numFreq
e.g., the original is N x M1 x M2
the resulting would be N' x M1 X M1
'''
def reshapeInputTensor(data_tensor):
    data_tensor_reshaped = []
    numSample = np.size(data_tensor, 0)
    numFreq   = np.size(data_tensor, 1)
    numBlock  = np.size(data_tensor, 2)

    numSubBlock = int(np.floor(np.divide(numBlock, numFreq)))
    numBlock_mod = int(np.multiply(numSubBlock, numFreq))
    data_tensor  = data_tensor[:, :, 0:numBlock_mod]
    for i in range(0, numSample):
        print(i)
        cur = data_tensor[i, :, :]
        tmp = np.reshape(cur, (numSubBlock, numFreq, numFreq))
        if len(data_tensor_reshaped) == 0:
            data_tensor_reshaped = tmp
        else:
            data_tensor_reshaped = np.concatenate((data_tensor_reshaped, tmp), axis=0)

    print(np.shape(data_tensor))
    print(np.shape(data_tensor_reshaped))
    return data_tensor_reshaped

'''
Average the activation map across both time and freq
input: 
    data_tensor: tensor, numSample x ch x numFreq x numBlock
output:
    feature_mat: array, numSample x ch
'''
def averageActivationMap(data_tensor):
    numSample, ch, numFreq, numBlock = np.shape(data_tensor)
    feature_mat = np.zeros((numSample, ch))
    for i in range(0, numSample):
        for j in range(0, ch):
            current_mat = data_tensor[i, j, :, :]
            feature_mat[i, j] = np.mean(current_mat)
    return feature_mat


'''
Track-wise z-score standardization on tensor
input:
    data_tensor: tensor, numSample x numFreq x numBlock
output:
    data_tensor: normalized tensor, same dimensionality as input
'''
def standardizeTensorTrackwise(data_tensor):
    numSample, numFreq, numBlock = np.shape(data_tensor)
    for i in range(0, numSample):
        avg = np.mean(data_tensor[i, :, :])
        std = np.std(data_tensor[i, :, :])
        if std != 0:
            data_tensor[i, :, :] = np.divide(data_tensor[i, :, :] - avg, std)
        else:
            data_tensor[i, :, :] = 0 * data_tensor[i, :, :]
    return data_tensor

'''
z-score standardization on tensor (entirely)
input:
    data_tensor: tensor, numSample x numFreq x numBlock
output:
    data_tensor: normalized tensor, same dimensionality as input
'''
def standardizeEntireTensor(data_tensor):
    numSample, numFreq, numBlock = np.shape(data_tensor)
    avg = np.mean(data_tensor)
    std = np.std(data_tensor)
    if std != 0:
        data_tensor = np.divide(data_tensor - avg, std)
    else:
        print('something is wrong!')
    return data_tensor


'''
Track-wise min-max scaling on tensor
input:
    data_tensor: tensor, numSample x numFreq x numBlock
output:
    data_tensor: scaled tensor, same dimensionality as input
'''
def scaleTensorTrackwise(data_tensor):
    numSample, numFreq, numBlock = np.shape(data_tensor)
    for i in range(0, numSample):
        minVal = np.min(data_tensor[i, :, :])
        maxVal = np.max(data_tensor[i, :, :])
        if (maxVal - minVal) != 0:
            data_tensor[i, :, :] = np.divide(data_tensor[i, :, :] - minVal, maxVal-minVal)
        else:
            data_tensor[i, :, :] = 0 * data_tensor[i, :, :]
    return data_tensor

'''
Track-wise L1 norm normalization on tensor
input:
    data_tensor: tensor, numSample x numFreq x numBlock
output:
    data_tensor: normalized tensor, same dimensionality as input
'''
def normalizeTensorTrackwiseL1(data_tensor):
    numSample, numFreq, numBlock = np.shape(data_tensor)
    for i in range(0, numSample):
        l1_norm = np.sum(abs(data_tensor[i, :, :]))
        if l1_norm != 0:
            data_tensor[i, :, :] = np.divide(data_tensor[i, :, :], l1_norm)
        else:
            data_tensor[i, :, :] = 0 * data_tensor[i, :, :]
    return data_tensor

def convert2dB(data_tensor):
    numSample, numFreq, numBlock = np.shape(data_tensor)
    for i in range(0, numSample):
        X = data_tensor[i, :, :]
        X = 10 * np.log10(np.maximum(X, 10e-6))
        X =  X - np.max(X)
        X = np.maximum(X, -80)
        data_tensor[i, :, :] = X
    return data_tensor