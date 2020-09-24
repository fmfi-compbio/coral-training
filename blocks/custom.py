import tensorflow as tf
from .util import seq, tcsconv, activations

class DPD(tf.keras.layers.Layer):
    def __init__(self, gaussian_noise=0, *, repeat, filters, kernel, activation, stride=1, residual=False, separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kwargs["kernel"]),
                    strides=(1, 1),
                    padding="same",
                    depth_multiplier=2, 
                    dilation_rate=(1, kwargs["dilation"]),
                ),
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=kwargs["out_channels"],
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]
        
        layers = []
        chan = in_channels
        for _ in range(repeat-1):
            layers.extend(
                _tcs(
                    in_channels=chan,
                    out_channels=filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1
                )
            )
            chan=filters
            layers.extend(
                _activation()
            )

        layers.extend(
            _tcs(
                in_channels=chan,
                out_channels=filters,
                kernel=kernel,
                separable=separable,
                stride=stride,
                bn_momentum=bn_momentum,
                dilation=1,
            )
        )
        self.head = layers
        
        if residual:
            self.residual = _tcs(
                in_channels=in_channels,
                out_channels=filters,
                kernel=kernel,
                stride=stride,
                separable=separable,
                bn_momentum=bn_momentum,
                dilation=1
            )
        else:
            self.residual = None

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        if self.residual:
            x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

BLOCKS = {
    "dpd": DPD,
}