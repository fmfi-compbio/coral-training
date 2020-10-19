import tensorflow as tf
import numpy as np

from blocks import Block
from blocks.util import seq, tcsconv

from tensorflow.python.ops import ctc_ops





class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='ctc_loss', smooth=0):
        super().__init__(reduction=reduction, name=name)
        self.smooth = smooth

    def call(self, y_label, y_pred):
        # should have been int32 in the first place but we can't pass different dtype for y
        y_label = tf.cast(y_label, tf.int32)

        y_logit = tf.nn.log_softmax(y_pred)
         
        input_length = tf.ones(shape=(tf.shape(y_pred)[0],), dtype=tf.int32) * tf.shape(y_pred)[1]
        label_length = tf.math.count_nonzero(y_label+1, 1, dtype=tf.int32)

        sparse_labels = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_label, label_length), tf.int32)

        # Note(ppershing): prefer explicit tf internals to never-ending call-stack of ctc_xyz delegations
        # between keras and tensorflow (and they fail to use cudnn anyway)
        ctc_loss = ctc_ops._ctc_loss_impl(
            labels=sparse_labels,
            inputs=y_logit,
            sequence_length=input_length,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True,
            ignore_longer_outputs_than_inputs=False,
            time_major=False,
            use_cudnn=True
        )

        #smooth_loss = - tf.math.reduce_sum(y_logit * [0.05, 0.05, 0.05, 0.05, 0.8], axis=-1)
        return tf.expand_dims(ctc_loss, axis=1) #+ self.smooth * smooth_loss

class Net(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        layers = []

        for layer_cfg in config:
            layers.append(
                Block(**layer_cfg)
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