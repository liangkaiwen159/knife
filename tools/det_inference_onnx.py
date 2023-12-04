from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmdet/instance-seg/instance-seg_onnxruntime_dynamic.py'
model_cfg = '../icann_dino_detr/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_knife.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmdet/ort/end2end.onnx']
image = 'demo/knife/knife.png'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_detection.png')