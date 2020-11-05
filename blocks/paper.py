import tensorflow as tf
import numpy as np

from .util import seq, activations, tcsconv, conv_init_identity, dw_init_identity, pw_init_identity, conv_init_diag, pool_init_identity, unpool_init_identity, conv_init_dw_identity

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

def space_to_depth(*, s2d, s2d_filters, **kwargs):
    return tf.keras.layers.Conv2D(
        filters=s2d_filters,
        kernel_size=(1, s2d),
        strides=(1, s2d),
        padding="same",
        **kwargs
    )

def depth_to_space(*, d2s, filters, **kwargs):
    return tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(1, d2s),
        strides=(1, d2s),
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
        ]
        if kernel > k:
            self.layers.append(
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel // k),
                    dilation_rate=(1, 3),
                    padding="same",
                    **dw_args,
                )
            )    

    def call(self, inputs, training=False):
        return seq(self.layers, inputs, training=training)

def batchnorm(**kwargs):
    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3, **kwargs)

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


# Bonito B-style residual block
class BlockBWithInit(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, kernel, **kwargs):
        super().__init__(**kwargs)
        
        layers = []
        for i in range(repeat):
            layers.extend([
                tf.keras.layers.SeparableConv2D(
                    filters=filters,
                    kernel_size=(1, kernel),
                    padding="same",
                    pointwise_initializer=conv_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                batchnorm(),
                activ() if i != repeat - 1 else None
            ])

        self.main = layers
        
        self.residual = [
            pointwise(filters=filters, kernel_initializer=conv_init_identity),
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

def _bn_get_scale_bias(bn_layer):
    # interpret batchnorm as x -> (x * scale + bias)
    import numpy as np

    gamma, beta, moving_mean, moving_variance = bn_layer.get_weights()
    scale = gamma / np.sqrt(moving_variance + 1e-3)
    bias = beta - moving_mean * gamma / np.sqrt(moving_variance  + 1e-3)
    return scale, bias
    
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


class BlockS2D(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, kernel, separable=False, **kwargs):
        super().__init__(**kwargs)

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tcsconv(
                    out_channels=s2d_filters,
                    kernel=kernel,
                    separable=separable,
                ),
                batchnorm(),
                activ(),
            ])

        layers.extend([
            depthwise(kernel=kernel),
            depth_to_space(
                d2s=s2d,
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


class BlockS2DInit(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, kernel, separable=False, **kwargs):
        super().__init__(**kwargs)

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.SeparableConv2D(
                    filters=s2d_filters,
                    kernel_size=(1, kernel),
                    padding="same",
                    pointwise_initializer=conv_init_identity,
                    depthwise_initializer=dw_init_identity,
                ),
                batchnorm(),
                activ(),
            ])

        layers.extend([
            depthwise(
                kernel=kernel,
                depthwise_initializer=dw_init_identity,
            ),
            depth_to_space(
                d2s=s2d,
                filters=filters,
            ),
            batchnorm(),
        ])
        self.head = layers
        
        self.residual = [
            pointwise(filters=filters, kernel_initializer=conv_init_identity),
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
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
                    kernel=kernel,
                    k=k,
                ),
                batchnorm(),
                activ(),
            ])

        layers.extend([
            depthwise(kernel=kernel),
            depth_to_space(
                d2s=s2d,
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
        _fold_bn(self.main, self.residual)


class BlockBothWithInit(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
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
            depth_to_space(
                d2s=s2d,
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
        _fold_bn(self.main, self.residual)


class BlockBothWithInit2(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
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
            depth_to_space(
                d2s=s2d,
                filters=filters,
                kernel_initializer="zeros"
            ),
            #batchnorm(gamma_initializer="zeros")
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
        _fold_bn(self.main, self.residual)


class BlockBothWithInit3(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
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
            depth_to_space(
                d2s=s2d,
                filters=filters,
            ),
            batchnorm()
        ])
        self.main = layers
        
        self.residual = [
            pointwise(filters=filters), #, kernel_initializer=conv_init_identity),
            batchnorm(),
        ]

        self.final_activation = activ()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return self.final_activation(x1 + x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)



class BlockBothWithInit4(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
                    kernel=kernel,
                    k=k,
                    #pw_args={"kernel_initializer": conv_init_identity},
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
            depth_to_space(
                d2s=s2d,
                filters=filters,
            ),
            batchnorm()
        ])
        self.main = layers
        
        self.residual = [
            pointwise(filters=filters), #, kernel_initializer=conv_init_identity),
            batchnorm(),
        ]

        self.final_activation = activ()
    
    def call(self, inputs, training=False):
        x1 = seq(self.main, inputs, training=training)
        x2 = seq(self.residual, inputs, training=training)
        return self.final_activation(x1 + x2, training=training)

    def fold_bn(self):
        _fold_bn(self.main, self.residual)

class BlockBothWithInit5(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
                    kernel=kernel,
                    k=k,
                    pw_args={"kernel_initializer": conv_init_dw_identity},
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
            depth_to_space(
                d2s=s2d,
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
        _fold_bn(self.main, self.residual)


class BlockBothWithInitA(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
                kernel_initializer=pool_init_identity,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
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
            depth_to_space(
                d2s=s2d,
                filters=filters,
                kernel_initializer=unpool_init_identity,
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
        _fold_bn(self.main, self.residual)


class BlockBothWithInitB(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
                kernel_initializer=pool_init_identity,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
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
            depth_to_space(
                d2s=s2d,
                filters=filters,
                #kernel_initializer="zeros",
            ),
            #batchnorm(gamma_initializer=lambda shape, dtype: tf.constant(np.ones(shape) * 1e-12, dtype=dtype))
            batchnorm(gamma_initializer="zeros")
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
        _fold_bn(self.main, self.residual)




class BlockBothWithInitC(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, k, kernel, **kwargs):
        super().__init__(**kwargs)
        assert kernel % k == 0

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                KSeparableConv(
                    filters=s2d_filters,
                    kernel=kernel,
                    k=k,
                    pw_args={"kernel_initializer": conv_init_identity},
                    #dw_args={"depthwise_initializer": dw_init_identity},
                ),
                batchnorm(),
                activ(),
            ])

        layers.extend([
            depthwise(
                kernel=kernel,
                #depthwise_initializer=dw_init_identity,
            ),
            depth_to_space(
                d2s=s2d,
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
        _fold_bn(self.main, self.residual)

class BlockSpecialWithInit(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, kernel, special_filters, **kwargs):
        super().__init__(**kwargs)

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.Conv2D(
                    filters=special_filters,
                    kernel_size=(1, kernel),
                    padding="same",
                    kernel_initializer=conv_init_identity,
                ),
                tf.keras.layers.Conv2D(
                    filters=s2d_filters,
                    kernel_size=(1, kernel),
                    dilation_rate=(1, kernel),
                    padding="same",
                    kernel_initializer=conv_init_identity,
                ),
                batchnorm(),
                activ(),
            ])

        layers.extend([
            depthwise(
                kernel=kernel**2,
                depthwise_initializer=dw_init_identity,
            ),
            depth_to_space(
                d2s=s2d,
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
        _fold_bn(self.main, self.residual)

class BlockSpecial2WithInit(tf.keras.layers.Layer):
    def __init__(self, *, repeat, filters, s2d_filters, s2d, kernel, dilation, final_kernel, special_filters, **kwargs):
        super().__init__(**kwargs)

        layers = [
            space_to_depth(
                s2d=s2d,
                s2d_filters=s2d_filters,
            ),
            batchnorm(),
            activ(),
        ]

        for _ in range(repeat-2):
            layers.extend([
                tf.keras.layers.Conv2D(
                    filters=special_filters,
                    kernel_size=(1, kernel[0]),
                    dilation_rate=(1, dilation[0]),
                    padding="same",
                    kernel_initializer=conv_init_dw_identity,
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(1, kernel[1]),
                    dilation_rate=(1, dilation[1]),
                    padding="same",
                    depthwise_initializer=dw_init_identity,
                ),
                tf.keras.layers.Conv2D(
                    filters=s2d_filters,
                    kernel_size=(1, kernel[2]),
                    dilation_rate=(1, dilation[2]),
                    padding="same",
                    kernel_initializer=conv_init_dw_identity,
                ),
                batchnorm(),
                activ(),
                tf.keras.layers.Dropout(0.05)
            ])

        layers.extend([
            depthwise(
                kernel=final_kernel,
                depthwise_initializer=dw_init_identity,
            ),
            depth_to_space(
                d2s=s2d,
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
        _fold_bn(self.main, self.residual)


BLOCKS = {
    "paperC": BlockC,
    "paperB": BlockB,
    "paperBInit": BlockBWithInit,

    "paperS2D": BlockS2D,
    "paperS2DInit": BlockS2DInit,
    "paperKSep": BlockKSep,
    "paperBoth": BlockBoth,
    "paperBothInit": BlockBothWithInit,
    "paperBothInit2": BlockBothWithInit2,
    "paperBothInit3": BlockBothWithInit3,
    "paperBothInit4": BlockBothWithInit4,
    "paperBothInit5": BlockBothWithInit5,


    "paperBothInitA": BlockBothWithInitA,
    "paperBothInitB": BlockBothWithInitB,
    "paperBothInitC": BlockBothWithInitC,


    "special": BlockSpecialWithInit,
    "special2": BlockSpecial2WithInit,

}