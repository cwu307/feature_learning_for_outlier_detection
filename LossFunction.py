from keras import backend as K
import tensorflow as tf
import numpy as np

_EPSILON = K.epsilon()
_ENTROPY_DROPOUT_RATIO = 0.05


def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)


def cross_entropy_max_dropout(y_true, y_pred):
    y_pred /= tf.reduce_sum(y_pred,
                            reduction_indices=len(y_pred.get_shape()) - 1,
                            keep_dims=True)
    # manual computation of crossentropy
    epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = y_true * tf.log(y_pred)
    # least_cross_entropy = -tf.nn.top_k(-cross_entropy, k=int(int(y_pred.get_shape()[0]) * (1.0-_ENTROPY_DROPOUT_RATIO))).values
    least_cross_entropy = -tf.nn.top_k(-cross_entropy, k=int(int(y_pred.get_shape()[1]) * (1.0-_ENTROPY_DROPOUT_RATIO))).values
    return -tf.reduce_sum(least_cross_entropy,
                          reduction_indices=len(y_pred.get_shape()) - 1)


def cross_entropy_max_dropout_np(y_true, y_pred):
    """For testing."""
    y_pred /= np.sum(y_pred)
    y_pred = np.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    cross_entropy = y_true * np.log(y_pred)
    least_cross_entropy = np.sort(cross_entropy)[:y_pred.size]
    print(least_cross_entropy)
    return -np.sum(least_cross_entropy)


if __name__ == '__main__':
    """Test"""
    yt = np.random.random([100])
    yp = np.random.random([100])

    np_result = cross_entropy_max_dropout_np(yt, yp)
    tf_result = K.eval(cross_entropy_max_dropout(K.variable(yt), K.variable(yp)))

    print(np_result)
    print(tf_result)

