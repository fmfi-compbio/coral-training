import tensorflow as tf
import numpy as np

from .util import seq, activations, tcsconv

class BlockC(tf.keras.layers.Layer):
    def __init__(self, *, separable, filters, kernel, stride=1, activation, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)

        self.layers = [
                tcsconv(out_channels=filters, kernel=kernel, stride=stride, separable=separable),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                tf.keras.layers.Activation(activations[activation]),
        ]

    def call(self, inputs, training=False):
        return seq(self.layers, inputs, training=training)


class BlockB(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, kernel, activation, stride=1, separable=False, dilation=1, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]
        
        layers = []
        for _ in range(repeat-1):
            layers.extend(
                _tcs(
                    out_channels=filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                )
            )
            layers.extend(
                _activation()
            )

        layers.extend(
            _tcs(
                out_channels=filters,
                kernel=kernel,
                separable=separable,
                stride=stride,
                bn_momentum=bn_momentum,
            )
        )
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
        )

        self.final = _activation()

   
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1 + x2, training=training)

class Block(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, kernel, activation, stride=1,residual=False,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = []
        for _ in range(repeat-1):
            layers.extend(
                _tcs(
                    out_channels=filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                )
            )
            layers.extend(
                _activation()
            )

        layers.extend(
            _tcs(
                out_channels=filters,
                kernel=kernel,
                separable=separable,
                stride=stride,
                bn_momentum=bn_momentum,
            )
        )
        self.main = layers
        
        if residual:
            self.residual = _tcs(
                out_channels=filters,
                kernel=kernel,
                stride=stride,
                separable=separable,
                bn_momentum=bn_momentum,
            )
        else:
            self.residual = None

        self.final = _activation()

    
    def call(self, inputs, training=False):
        x = seq(self.main, inputs, training=training)
        if self.residual:
            x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)


BLOCKS = {
    "default": Block,
    "blockB": BlockB,
    "blockC": BlockC,
}