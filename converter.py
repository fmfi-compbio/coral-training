import tensorflow as tf

def convert(model, get_batch, STEPS=5000, BATCH_SIZE=4):

    def representative_dataset_gen():
        for _ in range(100):
            x, y = get_batch(batch_size=BATCH_SIZE, leng=STEPS)
            yield [x]

    run_model = tf.function(lambda x: model(x))

    # This is important, let's fix the input size.
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([BATCH_SIZE, STEPS, 1], dtype=tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    return tflite_quant_model