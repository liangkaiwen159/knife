_base_ = [
    '../_base_/base_instance-seg_static.py',
    '../../_base_/backends/tensorrt.py'
]

onnx_config = dict(input_shape=(1280, 768))
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 768, 1280],
                    opt_shape=[1, 3, 768, 1280],
                    max_shape=[1, 3, 768, 1280])))
    ])