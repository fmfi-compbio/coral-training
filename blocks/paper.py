import tensorflow as tf
import numpy as np

from .util import seq, activations, tcsconv, conv_init_identity, dw_init_identity, pw_init_identity, conv_init_diag

def pointwise(*, filters, **kwargs):
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        padding="same",
        **kwargs
    )

def depthwise(*, kernel, **kwargs):
    return tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1, kernel),
        strides=(1, 1),
        padding="same",
        **kwargs
    )

def d2s(*, depth, d2s_filters, **kwargs):
    return tf.keras.layers.Conv2D(
        filters=d2s_filters,
        kernel_size=(1, depth),
        strides=(1, depth),
        padding="same",
        **kwargs
    )

def s2d(*, depth, filters, **kwargs):
    return tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(1, depth),
        strides=(1, depth),
        use_bias=False, #TFLITE does not have bias so no point in using it here
        **kwargs
    )

class KSeparableConv(tf.keras.layers.Layer):
    def __init__(self, *, filters, kernel, k, pw_args={}, dw_args={}, **kwargs):
        super().__init__(**kwargs)

        assert kernel % k == 0
        self.layers = [
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1, k),
                padding="same",
                **pw_args,
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, kernel // k),
                dilation_rate=(1, 3),
                padding="same",
                **dw_args,
            ),
        ]

    def call(self, inputs, training=False):
        return seq(self.layers, inputs, training=training)

def batchnorm():
    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)

def activ():
    return tf.keras.layers.Activation(activations["relu6"])

# Bonito C-style block
class BlockC(tf.keras.layers.Layer):
    def __init__(self, *, separable, filters, kernel, stride=1, **kwargs):
        super().__init__(**kwargs)

        self.layers = [
            tcsconv(out_channels=filters, kernel=kernel, stride=stride, separable=separable),
            batchnorm(),
            activ(),
        ]

    def call(self, inputs, training=False):
        return seq(self.layers, inputs, training=training)

# Bonito B-style residual block
class BlockB(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, kernel, separable=False, **kwargs):
        super().__init__(**kwargs)
        
        layers = []
        for i in range(repeat):
            layers.extend([
                tcsconv(
                    out_channels=filters,
                    kernel=kernel,
                    separable=separable,
                ),
                batchnorm(),
                activ() if i != repeat - 1 else None
            ])

        self.main = layers
        
        self.residual = [
            pointwise(filters=filters),
            batchnorm()
        ]

        self.final_activation = activ()

   
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return self.final_activation(x1 + x2, training=training)



class BlockKSep(tf.keras.layers.Layer):
    def __init__(self, *, repeat, k, filters, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0
           
        layers = []
        for i in range(repeat):
            layers.extend([
                KSeparableConv(
                    filters=filters,
                    kernel=kernel,
                    k=k,
                ),
                batchnorm(),
                activ() if i != repeat - 1 else None
            ])

        self.main = layers
        
        self.residual = [
            pointwise(filters=filters),
            batchnorm()
        ]

        self.final_activation = activ()

   
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return self.final_activation(x1 + x2, training=training)


def _fold_bn(main, residual):
    """
        Fold batchnorm after TransposeConv in Depth-to-Space residual blocks into residual branch.
        We do this because TransposeConv does not have bias term
    """
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


class BlockD2S(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, d2s_filters, depth, kernel, separable=False, **kwargs):
        super().__init__(**kwargs)

        layers = [
            d2s(
                depth=depth,
                d2s_filters=d2s_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tcsconv(
                    out_channels=d2s_filters,
                    kernel=kernel,
                    separable=separable,
                ),
                batchnorm(),
                activ(),
            ])

        layers.extend([
            depthwise(kernel=kernel),
            s2d(
                depth=depth,
                filters=filters,
            ),
            batchnorm(),
        ])
        self.head = layers
        
        self.residual = [
            pointwise(filters=filters),
            batchnorm()
        ]

        self.final_activation = activ()
    
    def call(self, inputs, training=False):
        x1 = seq(self.head, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return self.final_activation(x1 + x2, training=training)

    def fold_bn(self):
        _fold_bn(self.head, self.residual)


class BlockBoth(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, d2s_filters, depth, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            d2s(
                depth=depth,
                d2s_filters=d2s_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=d2s_filters,
                    kernel=kernel,
                    k=k,
                ),
                batchnorm(),
                activ(),
            ])

        layers.extend([
            depthwise(kernel=kernel),
            s2d(
                depth=depth,
                filters=filters,
            ),
            batchnorm()
        ])
        self.main = layers
        
        self.residual = [
            pointwise(filters=filters),
            batchnorm(),
        ]

        self.final_activation = activ()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return self.final_activation(x1 + x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main)


class BlockBothWithInit(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, d2s_filters, depth, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            d2s(
                depth=depth,
                d2s_filters=d2s_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=d2s_filters,
                    kernel=kernel,
                    k=k,
                    pw_args={"kernel_initializer": conv_init_identity},
                    dw_args={"depthwise_initializer": dw_init_identity},
                ),
                batchnorm(),
                activ(),
            ])

        layers.extend([
            depthwise(
                kernel=kernel,
                depthwise_initializer=dw_init_identity,
            ),
            s2d(
                depth=depth,
                filters=filters,
            ),
            batchnorm()
        ])
        self.main = layers
        
        self.residual = [
            pointwise(filters=filters, kernel_initializer=conv_init_identity),
            batchnorm(),
        ]

        self.final_activation = activ()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return self.final_activation(x1 + x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main)


BLOCKS = {
    "paperC": BlockC,
    "paperB": BlockB,
    "paperD2S": BlockD2S,
    "paperKSep": BlockKSep,
    "paperBoth": BlockBoth,
    "paperBothInit": BlockBothWithInit,
}