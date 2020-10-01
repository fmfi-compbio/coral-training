import tensorflow as tf

def Decoder(init="default"):
    _init = {
        "default": dict(
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ),
        "zeros": dict(
            kernel_initializer="zeros",
            bias_initializer=lambda shape, dtype=None: tf.constant([0, 0, 0, 0, 2], dtype=dtype)
        )
    }[init]

    return tf.keras.layers.Conv2D(
        filters=5,
        kernel_size=(1, 1),
        padding="same",
        **_init
    )

BLOCKS = {
    "decoder": Decoder,
}