import tensorflow as tf

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        y_true = tf.cast(y_true, tf.int32)
         
        input_length = tf.ones(shape=(tf.shape(y_pred)[0], 1), dtype=tf.int32) * tf.shape(y_pred)[1]
        label_length = tf.math.count_nonzero(y_true+1, 1, keepdims=True)

        return tf.keras.backend.ctc_batch_cost(
            y_true,
            y_pred,
            input_length,
            label_length
        )


def tcsconv(*, in_channels, out_channels, kernel, stride, separable):
    if separable:
        return tf.keras.layers.SeparableConv2D(
            filters=out_channels,
            kernel_size=(1, kernel),
            strides=(1, stride),
            padding="same",
        )
    else:
        return tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=(1, kernel),
            strides=(1, stride),
            padding="same",
        )


def seq(layers, inputs, **kwargs):
    for l in layers:
        inputs = l(inputs, **kwargs)
    return inputs

activations = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,

}

class Block(tf.keras.layers.Layer):
    def __init__(self, *, repeat,in_channels,filters,kernel, activation, dropout=0.0,stride=1,residual=False,separable=False, bn_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.config = {
            "repeat": repeat,
            "in_channels": in_channels,
            "filters": filters,
            "kernel": kernel,
            "dropout": dropout,
            "stride": stride,
            "residual": residual,
            "separable": separable,
            "activation": activation,
            "bn_momentum": bn_momentum,
        }
        
        def _tcs(*, bn_momentum, **kwargs):
            return [
                tcsconv(**kwargs),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=1e-3)
            ]

        def _activation():
            return [
                tf.keras.layers.Activation(activations[activation]),
                tf.keras.layers.Dropout(dropout)
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
            )
        else:
            self.residual = None

        self.final = _activation()

    def get_config(self):
        base_config = super().get_config()

        return {
            **base_config,
            **self.config
        }
    
    def call(self, inputs, training=False):
        x = seq(self.head, inputs, training=training)
        if self.residual:
            x += seq(self.residual, inputs, training=training)
        return seq(self.final, x, training=training)

def Decoder():
    return tf.keras.layers.Conv2D(
        filters=5,
        kernel_size=(1, 1),
        padding="same"
    )
    
class Net(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        channels = 1

        layers = []

        for layer_cfg in config:
            layers.append(
                Block(
                    in_channels=channels, **layer_cfg
                )
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