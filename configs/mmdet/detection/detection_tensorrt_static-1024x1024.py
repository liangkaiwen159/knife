_base_ = ['../_base_/base_tensorrt_static-800x1344.py']

onnx_config = dict(input_shape=(1024, 1024))

backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 1024, 1024],
                max_shape=[1, 3, 1024, 1024],
                opt_shape=[1, 3, 1024, 1024],
                default_shape=[1, 3, 1024, 1024])))
])
