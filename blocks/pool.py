import tensorflow as tf
from .util import seq, tcsconv, activations, pw_init_identity, dw_init_identity

def _fold_bn(main, residual):
    import numpy as np

    gamma, beta, moving_mean, moving_variance = main[-1].get_weights()
    scale = gamma / np.sqrt(moving_variance + 1e-3)
    bias = beta - moving_mean * gamma / np.sqrt(moving_variance  + 1e-3)

    w, = main[-2].get_weights()
    scale = scale.reshape((1, 1, -1, 1))
    main[-2].set_weights([w * scale])

    gamma, beta, moving_mean, moving_variance = residual[-1].get_weights()
    residual[-1].set_weights([gamma, beta + bias, moving_mean, moving_variance])
    main.pop()

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
        _fold_bn(self.head, self.residual)


class PoolConv(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
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

        self.final = [
            *_activation(),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ]
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


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

class PoolXInitIpre(tf.keras.layers.Layer):
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
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, gamma_initializer="zeros"),
            *_activation()
        ])
        self.head = layers
        
        self.residual = _tcs(
            out_channels=filters,
            kernel=kernel,
        )

        #self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return x #seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)

class PoolXInitIpre2(tf.keras.layers.Layer):
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
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, gamma_initializer="zeros"),
            *_activation()
        ])
        self.head = layers
        
        self.residual = [
            *_tcs(
                out_channels=filters,
                kernel=kernel,
            ),
            *_activation()
        ]

    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x += seq(self.residual, inputs, training=training)
        return x #seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)

class PoolXInitIpre3(tf.keras.layers.Layer):
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
                depthwise_initializer=dw_init_identity,
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3, gamma_initializer="zeros"),
            *_activation()
        ])
        self.head = layers
        
        self.residual = [
            *_tcs(
                out_channels=filters,
                kernel=kernel,
            ),
            *_activation()
        ]

    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        x -= seq(self.residual, inputs, training=training)
        return x #seq(self.final, x, training=training)

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

class PoolXY(tf.keras.layers.Layer):
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
        

        layers = []

        for _ in range(repeat):
            layers.extend([
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
                tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
                *_activation(),
            ])

        layers.pop()
        layers.append(
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
        )

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


class PoolXU(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, bn_momentum, **kwargs):
            return [
                tf.keras.layers.Conv2D(
                    filters=kwargs["out_channels"],
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kwargs["kernel"]),
                    strides=(1, 1),
                    padding="same",
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

class PoolXV(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, pool_kernel=None, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        pool_kernel = pool_kernel if pool_kernel else kernel

        def _tcs(*, bn_momentum, **kwargs):
            return [
                tf.keras.layers.Conv2D(
                    filters=kwargs["out_channels"],
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kwargs["kernel"]),
                    strides=(1, 1),
                    padding="same",
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

        def act(i):
            return [
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                *_activation(),                
            ] if i % 3 == 2 else []

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, pool_kernel),
                    strides=(1, 1),
                    padding="same",
                ),
                *act(_*2),
                tf.keras.layers.Conv2D(
                    filters=pool_filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                ),
                *act(_*2 + 1)
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
        if self.residual:
            x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolXDecor(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
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
                kernel_size=(1, kernel),
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
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
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

        self.final = [
            tf.keras.layers.Activation(activations[activation])
        ]
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
            x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolY(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
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
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
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


class PoolZ(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
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
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers

        self.pre = [
            tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), padding="same"),
        ]
        self.residual = [
            tf.keras.layers.DepthwiseConv2D(kernel_size=(1,kernel), strides=(1, 1), padding="same"),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ]

        self.final = _activation()
    
    def call(self, inputs, training=False):
        inputs = seq(self.pre, inputs, training=training)
        x1 = seq(self.head, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return seq(self.final, x1 + x2, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolF(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
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
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
            self.residual = [
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=filters // 2,
                ),      
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    padding="same",
                    depth_multiplier=2, 
                ),
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=filters,
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


class PoolFF(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters // 2,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
                depth_multiplier=2, 
            ),
            tf.keras.layers.Conv2D(
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                filters=pool_filters,
            ),            
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            tf.keras.layers.Activation(activations[activation]),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=pool_filters // 2,
                ),      
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    padding="same",
                    depth_multiplier=2, 
                ),
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=pool_filters,
                ),            
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                tf.keras.layers.Activation(activations[activation]),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
            self.residual = [
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=filters // 2,
                ),      
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    padding="same",
                    depth_multiplier=2, 
                ),
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=filters,
                ),            
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        self.final = [
            tf.keras.layers.Activation(activations[activation]),
        ]
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
            x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolJJ(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)

        layers = [
            tf.keras.layers.Conv2D(
                filters=pool_filters // 2,
                kernel_size=(1, pool),
                strides=(1, pool),
                padding="same",
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
                depth_multiplier=2, 
            ),       
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            tf.keras.layers.Activation(activations[activation]),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=pool_filters // 2,
                ),      
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    padding="same",
                    depth_multiplier=2, 
                ),       
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                tf.keras.layers.Activation(activations[activation]),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
            self.residual = [
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=filters // 2,
                ),      
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    padding="same",
                    depth_multiplier=2, 
                ),       
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        self.final = [
            tf.keras.layers.Activation(activations[activation]),
        ]
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
            x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class PoolG(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
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
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
            self.residual = [
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=filters // 2,
                ),      
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    padding="same",
                    depth_multiplier=2, 
                ),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
                tf.keras.layers.Activation(activations[activation]),
                tf.keras.layers.Conv2D(
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    filters=filters,
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


class PoolH(tf.keras.layers.Layer):
    def __init__(self, *, pool, pool_filters, repeat,filters,kernel, activation, stride=1,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        
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
            layers.extend([
                *_tcs(
                    out_channels=pool_filters,
                    kernel=kernel,
                    separable=separable,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    dilation=1 if _ == 1 else 1,
                ),
                *_activation(),
            ])

        layers.extend([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=(1, 1),
                padding="same",
            ),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, pool), strides=(1,pool), use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
        ])
        self.head = layers
        
            self.residual = [
                tf.keras.layers.Conv2D(
                    kernel_size=(1,2),
                    strides=(1,2),
                    padding="same",
                    filters=filters,
                ),      
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel),
                    strides=(1, 1),
                    padding="same",
                    depth_multiplier=1, 
                ),
                tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, 2), strides=(1,2), use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3),
            ]

        self.final = _activation()
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
            x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)



BLOCKS = {
    "pool": Pool,
    "poolc": PoolConv,
    "poolx": PoolX,
    "poolxbn": PoolXBN,

    "poolxiniti": PoolXInitI,

    "poolxinitj": PoolXInitJ,
    "poolxinitii": PoolXInitII,
    "poolxinitipre": PoolXInitIpre,
    "poolxinitipre2": PoolXInitIpre2,
    "poolxinitipre3": PoolXInitIpre3,



    "poolxu": PoolXU,
    "poolxv": PoolXV,
    "poolxy": PoolXY,

    "pooly": PoolY,
    "poolz": PoolZ,
    "poolf": PoolF,
    "poolff": PoolFF,
    "pooljj": PoolJJ,
    "poolg": PoolG,
    "poolh": PoolH,
}
