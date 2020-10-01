import tensorflow as tf
from .util import seq, tcsconv, activations, pw_init_identity, dw_init_identity, pool_init_identity, unpool_init_identity, conv_init_identity

from functools import wraps

BLOCKS = {}

def register(name):
    def decorator(block):
        BLOCKS[name] = block
        return block
    return decorator


def _bn_get_scale_bias(bn_layer):
    # interpret batchnorm as x -> (x * scale + bias)
    import numpy as np

    gamma, beta, moving_mean, moving_variance = bn_layer.get_weights()
    scale = gamma / np.sqrt(moving_variance + 1e-3)
    bias = beta - moving_mean * gamma / np.sqrt(moving_variance  + 1e-3)
    return scale, bias


def _fold_bn(main, residual):
    bn = main[-1]
    transp = main[-2]
    other_bn = residual[-1]

    scale, bias = _bn_get_scale_bias(bn)

    # fold scale into previous transposeconv
    w, = transp.get_weights()
    scale = scale.reshape((1, 1, -1, 1))
    transp.set_weights([w * scale])

    # fold bias into residual's batchnorm
    gamma, beta, moving_mean, moving_variance = other_bn.get_weights()
    other_bn.set_weights([gamma, beta + bias, moving_mean, moving_variance])
    
    # drop bn from graph
    main.pop()


def _fold_bn_conv2d(main):
    import numpy as np

    bn = main[0]
    conv = main[1]

    scale, bias = _bn_get_scale_bias(bn)
    
    # fold into subsequent conv2d
    w, b = conv.get_weights()
    scale = scale.reshape((1, 1, -1, 1))
    ww = w * scale

    bias = bias.reshape((1, 1, -1, 1))
    bb = b + tf.reduce_sum(w * bias, axis=[0,1,2])
    conv.set_weights([ww, bb])

    # drop bn from graph
    main.pop(0)


# assumes depthwise has 
# this is lossy
def _fold_bn_depthwise(main):
    import numpy as np

    bn = main[0]
    dw = main[1]

    scale, bias = _bn_get_scale_bias(bn)
    # fold into subsequent dw
    w, b = dw.get_weights()
    scale = scale.reshape((1, 1, -1, 1))
    print("SCALE", scale)
    ww = w * scale

    #print("BIAS", bias)
    bb = b + tf.reduce_sum(w, axis=[0,1,3]) * bias
    print("BIAS", tf.reduce_sum(w, axis=[0,1,3]) * bias)
    dw.set_weights([ww, bb])

    # drop bn from graph
    main.pop(0)


class Pool(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1, separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
        def _tcsbn(*, bn_momentum, **kwargs):
            return [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kwargs["kernel"]),
                    strides=(1, 1),
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, beta_initializer=tf.keras.initializers.Constant(3.0)),
                tf.keras.layers.Activation(activations[activation]),
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=kwargs["out_channels"],
                ),
            ]
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.MaxPool2D(pool_size=(1, pool)),
        ]
        for _ in range(repeat-1):
            layers.extend(
                _tcs(
                    out_channels=pool_filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                )
            )
            layers.extend(
                _activation()
            )

        layers.extend(
            _tcs(
                out_channels=pool_filters,
                kernel=kernel,
                separable=separable,
                stride=stride,
                bn_momentum=bn_momentum,
                dilation=1,
            )
        )
        layers.extend([
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            tf.keras.layers.Activation(activations[activation]),

            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1, pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)


class PoolEraseA(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
        ]

        for _ in range(repeat-2):
            layers.extend(
                _tcs(
                    out_channels=pool_filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                )
            )
            layers.extend(
                _activation() if _ % 2 == 0 else [tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)]
            )

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)


class PoolEraseB(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.SeparableConv2D(
                    filters=pool_filters,
                    kernel_size=(1, pool_kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            ])
            layers.extend(
                _activation() if _ % 2 == 0 else []
            )

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)


class PoolEraseC(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),

            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.SeparableConv2D(
                    filters=pool_filters,
                    kernel_size=(1, pool_kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)

class PoolEraseCInit(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, separable="whatever", bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),

            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.SeparableConv2D(
                    filters=pool_filters,
                    kernel_size=(1, pool_kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, gamma_initializer="zeros")
        ])
        self.main = layers
        
        self.residual = [
            tf.keras.layers.SeparableConv2D(
                filters=filters,
                kernel_size=(1, kernel),
                padding="same",
                pointwise_initializer=pw_init_identity,
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ]

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)


@register("poolerased")
class PoolEraseD(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),

            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.SeparableConv2D(
                    filters=pool_filters,
                    kernel_size=(1, pool_kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)

@register("poolerasedx")
class PoolEraseDX(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),

            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.Conv2D(
                    filters=pool_filters,
                    kernel_size=(1, 1),
                    padding="same",
                    kernel_initializer=pw_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, pool_kernel),
                    padding="same",
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)


@register("poolerasedd")
class PoolEraseDD(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),

            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.Conv2D(
                    filters=pool_filters,
                    kernel_size=(1, 3),
                    padding="same",
                    kernel_initializer=conv_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, pool_kernel),
                    dilation_rate=(1, 3),
                    padding="same",
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)

@register("pooleraseddd")
class PoolEraseDDD(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),

            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.Conv2D(
                    filters=pool_filters,
                    kernel_size=(1, 3),
                    padding="same",
                    kernel_initializer=conv_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, pool_kernel),
                    dilation_rate=(1, 3),
                    padding="same",
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = [
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1, 3),
                padding="same",
                kernel_initializer=conv_init_identity,
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                dilation_rate=(1, 3),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
        ]

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)

@register("pooleraseedd")
class PoolEraseDDD(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),

            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.Conv2D(
                    filters=pool_filters,
                    kernel_size=(1, 3),
                    padding="same",
                    kernel_initializer=conv_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, pool_kernel),
                    dilation_rate=(1, 3),
                    padding="same",
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = [
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1, 3),
                padding="same",
                kernel_initializer=conv_init_identity,
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                dilation_rate=(1, 3),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
        ]

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)


@register("poolerasee")
class PoolEraseE(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),

            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.SeparableConv2D(
                    filters=pool_filters,
                    kernel_size=(1, pool_kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.main = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1+x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)



class PoolX(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
            dilation=1,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)

class PoolXD(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat, filters, kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, bn_momentum, **kwargs):
            return [
                tf.keras.layers.Conv2D(
                    filters=kwargs["out_channels"],
                    kernel_size=(1, 3),
                    strides=(1, 1),
                    padding="same",
                    #kernel_initializer=conv_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kwargs["kernel"]),
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
            ] if activation else []
        
        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
            stride=stride,
            separable=separable,
            bn_momentum=bn_momentum,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolXBN(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ] if activation else []
        
        layers = [
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
        self.residual = [
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, beta_initializer="zeros"),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1, 1),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
        ]
        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)
        _fold_bn_conv2d(self.head)
        _fold_bn_depthwise(self.residual)

class PoolXInitI(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, kernel, out_channels):
            return [
                tf.keras.layers.SeparableConv2D(
                    filters=out_channels,
                    kernel_size=(1, kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolXInitII(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, kernel, out_channels):
            return [
                tf.keras.layers.SeparableConv2D(
                    filters=out_channels,
                    kernel_size=(1, kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
                #kernel_initializer=pw_init_identity,
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(1, pool),
                strides=(1,pool),
                use_bias=False,
                kernel_initializer="zeros"
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)

class PoolXInitIII(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, kernel, out_channels):
            return [
                tf.keras.layers.SeparableConv2D(
                    filters=out_channels,
                    kernel_size=(1, kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
                #kernel_initializer=pw_init_identity,
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(1, pool),
                strides=(1,pool),
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, gamma_initializer="zeros")
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolXInitIV(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, kernel, out_channels):
            return [
                tf.keras.layers.SeparableConv2D(
                    filters=out_channels,
                    kernel_size=(1, kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
                kernel_initializer=pool_init_identity,
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(1, pool),
                strides=(1,pool),
                use_bias=False,
                kernel_initializer="zeros"
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolXInitV(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, kernel, out_channels):
            return [
                tf.keras.layers.SeparableConv2D(
                    filters=out_channels,
                    kernel_size=(1, kernel),
                    padding="same",
                    pointwise_initializer=pw_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
                kernel_initializer=pool_init_identity,
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(1, pool),
                strides=(1,pool),
                use_bias=False,
                kernel_initializer=unpool_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, gamma_initializer="zeros")
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolXInitIIBN(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, kernel, out_channels):
            return [
                tf.keras.layers.Conv2D(
                    filters=out_channels,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    #kernel_initializer=pw_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    padding="same",
                    #depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = [
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(1, pool),
                strides=(1,pool),
                use_bias=False,
                kernel_initializer="zeros"
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
        self.residual = [
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_tcs(
                out_channels=filters,
                kernel=kernel,
            )
        ]
        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)
        _fold_bn_conv2d(self.head)
        _fold_bn_conv2d(self.residual)


class PoolPD(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, kernel, out_channels):
            return [
                tf.keras.layers.Conv2D(
                    filters=out_channels,
                    kernel_size=(1, 1),
                    padding="same",
                    kernel_initializer=pw_init_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    padding="same",
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
                #kernel_initializer=pw_init_identity,
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=pool_kernel,
                ),
                *_activation(),
            ])

        layers.extend([
            #tf.keras.layers.DepthwiseConv2D(
            #    kernel_size=(1, pool_kernel),
            #    strides=(1, 1),
            #    padding="same",
            #    depthwise_initializer=dw_init_identity,
            #),
            tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(1, pool),
                strides=(1,pool),
                use_bias=False,
                kernel_initializer="zeros"
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
        )

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)

class PoolXInitJ(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, kernel, out_channels):
            return [
                tf.keras.layers.SeparableConv2D(
                    filters=out_channels,
                    kernel_size=(1, kernel),
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
            ]

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            *_activation(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.SeparableConv2D(
                    filters=pool_filters,
                    kernel_size=(1, pool_kernel),
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, pool_kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, gamma_initializer="zeros")
        ])
        self.head = layers
        
        self.residual = [
            tf.keras.layers.SeparableConv2D(
                filters=filters,
                kernel_size=(1, kernel),
                padding="same",
                pointwise_initializer=pw_init_identity,
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ]

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)




BLOCKS.update({
    "pool": Pool,
    "poolx": PoolX,
    "poolxd": PoolXD,

    "poolxbn": PoolXBN,

    "poolxiniti": PoolXInitI,

    "poolxinitj": PoolXInitJ,
    "poolxinitii": PoolXInitII,
    "poolxinitiii": PoolXInitIII,
    "poolxinitiv": PoolXInitIV,
    "poolxinitv": PoolXInitV,

    "poolxinitiibn": PoolXInitIIBN,

    "poolerasea": PoolEraseA,
    "pooleraseb": PoolEraseB,
    "poolerasec": PoolEraseC,
    "poolerasecinit": PoolEraseCInit,
    "poolerased": PoolEraseD,

})
