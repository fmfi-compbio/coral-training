import tensorflow as tf
import numpy as np

from blocks import Block
from blocks.util import seq, tcsconv

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


def Decoder():
    return tf.keras.layers.Conv2D(
        filters=5,
        kernel_size=(1, 1),
        padding="same",
        #bias_initializer=lambda shape, dtype=None: tf.constant([0, 0, 0, 0, 2], dtype=dtype)
    )

class Net(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        channels = 1

        layers = []

        for layer_cfg in config:
            layers.append(
                Block(**layer_cfg)
            )
            channels = layer_cfg['filters']

        layers.append(
            Decoder()
        )
        self.layers = layers
        self.config = config
        
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "config": self.config
        }
    
    def call(self, inputs, training=False):
        return seq(self.layers, inputs, training=training)
    

def make_model(cfg):
    input_data = tf.keras.Input(name='the_input', shape=(None, 1), dtype='float32')
    net = Net(cfg)
    y_pred = tf.squeeze(net(tf.expand_dims(input_data, axis=0)), axis=0)
    return tf.keras.Model(inputs=input_data, outputs=y_pred)