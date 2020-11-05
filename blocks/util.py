import tensorflow as tf
import numpy as np

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        y_true = tf.cast(y_true, tf.int32)
         
        input_length = tf.ones(shape=(tf.shape(y_pred)[0], 1), dtype=tf.int32) * tf.shape(y_pred)[1]
        label_length = tf.math.count_nonzero(y_true+1, 1, keepdims=True)

        ctc_loss = tf.keras.backend.ctc_batch_cost(
            y_true,
            y_pred,
            input_length,
            label_length
        )
        #smooth_loss = - tf.math.reduce_sum(tf.math.log(y_pred) * [0.05, 0.05, 0.05, 0.05, 0.8], axis=-1)
        return ctc_loss #+ smooth_loss * 41

class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(
            'bias',
            shape=input_shape[1:],
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        return x + self.bias

def dw_init_identity(shape, dtype=None):
    assert(len(shape) == 4)
    assert(shape[0] == 1)
    assert(shape[3] == 1)
    data = np.random.uniform(-0.01, 0.01, shape)
    for i in range(shape[2]):
        data[0][shape[1]//2][i][0] = 1.0
    return tf.constant(data, dtype=dtype)

def pw_init_identity(shape, dtype=None):
    assert(len(shape) == 4)
    assert(shape[0] == 1)
    assert(shape[1] == 1)
    data = np.random.uniform(-0.01, 0.01, shape)
    for i in range(shape[3]):
        data[0][0][i % shape[2]][i] = 1.0
    return tf.constant(data, dtype=dtype)

def conv_init_identity(shape, dtype=None):
    assert(len(shape) == 4)
    assert(shape[0] == 1)
    data = np.random.uniform(-0.01, 0.01, shape)  
    for i in range(shape[3]):
        data[0][shape[1]//2][i % shape[2]][i] = 1.0
    return tf.constant(data, dtype=dtype)


def conv_init_dw_identity(shape, dtype=None):
    assert(len(shape) == 4)
    assert(shape[0] == 1)
    data = np.random.uniform(-0.1, 0.1, shape)
    for i in range(shape[1]):
        if i == shape[1] // 2:
            continue
        for j in range(shape[2]):
            for k in range(shape[3]):
                data[0][i][j][k] = 0
    return tf.constant(data, dtype=dtype)


def conv_init_diag(shape, dtype=None):
    assert(len(shape) == 4)
    assert(shape[0] == 1)
    data = np.random.uniform(-0.01, 0.01, shape)
    for i in range(shape[3]):
        data[0][shape[1] // 2][i % shape[2]][i] = np.random.uniform(-1, 1)
    return tf.constant(data, dtype=dtype)


def pool_init_identity(shape, dtype=None):
    #print("SHAPE", shape)
    assert(len(shape) == 4)
    assert(shape[0] == 1)
    #assert(shape[1] == 1)
    data = np.random.uniform(-0.01, 0.01, shape)
    for j in range(shape[1]):
        for i in range(shape[3]):
            data[0][j][i % shape[2]][i] = 1.0
    return tf.constant(data, dtype=dtype)


def unpool_init_identity(shape, dtype=None):
    #print("SHAPE", shape)
    assert(len(shape) == 4)
    assert(shape[0] == 1)
    #assert(shape[1] == 1)
    data = np.random.uniform(-0.01, 0.01, shape)
    for j in range(shape[1]):
        for i in range(shape[3]):
            data[0][j][i % shape[2]][i] = 1.0
    return tf.constant(data, dtype=dtype)


def tcsconv(*, separable, out_channels, kernel, stride=1, dilation=1, activation=None):
    if separable:
        return tf.keras.layers.SeparableConv2D(
            filters=out_channels,
            kernel_size=(1, kernel),
            strides=(1, stride),
            padding="same",
            dilation_rate=(1, dilation),
            activation=activation,
        )
    else:
        return tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=(1, kernel),
            strides=(1, stride),
            padding="same",
            activation=activation,
        )


def seq(layers, inputs, **kwargs):
    for l in layers:
        if l is not None:
            inputs = l(inputs, **kwargs)
    return inputs

activations = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
}
