import tensorflow as tf
import numpy as np

from .util import seq, activations, tcsconv, conv_init_identity, dw_init_identity, pw_init_identity, conv_init_diag

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


class ConvInit(tf.keras.layers.Layer):
    def __init__(self, *, filters, kernel, activation, stride=1, bn_momentum=0.9, init="ident", **kwargs):
        super().__init__(**kwargs)

        kernel_initializer = {
            "ident": conv_init_identity,
            "diag": conv_init_diag,
        }[init]
        
        self.layers = [
                tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, kernel), strides=(1, stride), padding="same", kernel_initializer=kernel_initializer),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                tf.keras.layers.Activation(activations[activation]),
        ]

    def call(self, inputs, training=False):
        return seq(self.layers, inputs, training=training)


class SepInit(tf.keras.layers.Layer):
    def __init__(self, *, filters, kernel, activation, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)

        self.layers = [
                tf.keras.layers.DepthwiseConv2D(kernel_size=(1, kernel), depthwise_initializer=dw_init_identity, padding="same"),
                tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), kernel_initializer=pw_init_identity),
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
            kernel=1,
            stride=1,
            separable=False,
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


class BlockD(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, kernel, activation, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
        def _tcs(*, bn_momentum, kernel, filters):
            return [
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=(1, 3),
                    strides=(1, 1),
                    padding="same",
                    #kernel_initializer=pool_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    dilation_rate=(1, 3),
                    padding="same",
                    #depthwise_initializer=dw_init_identity,
                ),
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
                    filters=filters,
                    kernel=kernel,
                    bn_momentum=bn_momentum,
                )
            )
            layers.extend(
                _activation()
            )

        layers.extend(
            _tcs(
                filters=filters,
                kernel=kernel,
                bn_momentum=bn_momentum,
            )
        )
        self.main = layers
        
        self.residual = _tcs(
            filters=filters,
            kernel=kernel,
            bn_momentum=bn_momentum,
        )

        self.final = _activation()

   
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1 + x2, training=training)

class BlockDInit(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, kernel, activation, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
        def _tcs(*, bn_momentum, kernel, filters, gamma_initializer="ones"):
            return [
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=(1, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=conv_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    dilation_rate=(1, 3),
                    padding="same",
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, gamma_initializer=gamma_initializer)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]
        
        layers = []
        for _ in range(repeat-1):
            layers.extend([
                *_tcs(
                    filters=filters,
                    kernel=kernel,
                    bn_momentum=bn_momentum,
                ),
                *_activation()
            ])

        layers.extend(
            _tcs(
                filters=filters,
                kernel=kernel,
                bn_momentum=bn_momentum,
                gamma_initializer="zeros",
            )
        )
        self.main = layers
        
        self.residual = _tcs(
            filters=filters,
            kernel=kernel,
            bn_momentum=bn_momentum,
        )

        self.final = _activation()

   
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1 + x2, training=training)


BLOCKS = {
    "default": Block,
    "blockB": BlockB,
    "blockC": BlockC,
    "blockD": BlockD,
    "blockdinit": BlockDInit,
    "convinit": ConvInit,
    "sepinit": SepInit,
}