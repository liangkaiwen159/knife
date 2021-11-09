import tensorrt as trt

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=False, log_level=trt.Logger.INFO, max_workspace_size=0))