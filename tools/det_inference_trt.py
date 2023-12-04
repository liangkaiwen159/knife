from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch
import os
from tqdm import tqdm
import time
from det_inference_torch import infer_torch
from det_inference_torch import save_dir as torch_save_dir
import logging
import cv2
import numpy as np


logger = logging.getLogger("mmengine")

# 设置 logger 的日志级别
logger.setLevel(logging.ERROR)


    
deploy_cfg = 'configs/mmdet/detection/detection_tensorrt_static-1024x1024.py'
model_cfg = '../icann_dino_detr/configs/dino_rot/dino-4scale_dota.py'
device = 'cuda'
backend_model = ['./mmdeploy_models/dino/ort/end2end.engine']
image_dir = '/home/liangkaiwen/datasets/split_ss_dota/val/images/'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)

imgs = sorted(os.listdir(image_dir))[:100]

total_time = 0
total_num = len(imgs)

save_dir = 'output_tensorrt'

if os.path.exists(save_dir):
    os.system(f'rm -rf {save_dir}')
    os.mkdir(save_dir)
else:
    os.mkdir(save_dir)
    
def infer_trt():
    global total_time
    for img_name in tqdm(imgs):
        image = os.path.join(image_dir, img_name)
        t1 = time.time()
        model_inputs, _ = task_processor.create_input(image, input_shape)
        
    # do model inference
        with torch.no_grad():
            result = model.test_step(model_inputs)
        pred_instances = result[0].pred_instances
        keep_idx = pred_instances.scores > 0.5
        pred_instances = pred_instances[keep_idx]
        result[0].pred_instances = pred_instances
        # visualize results
        task_processor.visualize(
            image=image,
            model=model,
            result=result[0],
            window_name='visualize',
            output_file=f'{save_dir}/{img_name}')
        t2 = time.time()
        total_time += t2 - t1
    print(f'average trt time: {round(total_time / total_num, 3)}s')


if __name__ == '__main__':
    infer_trt()
    infer_torch()
    trt_save_dir = save_dir
    img_list = sorted(os.listdir(trt_save_dir))
    compare_dir = 'compare'
    if os.path.exists(compare_dir):
        os.system(f'rm -rf {compare_dir}')
        os.mkdir(compare_dir)
    else:
        os.mkdir(compare_dir)
    for img_name in tqdm(img_list):
        img1 = os.path.join(torch_save_dir, 'vis', img_name)
        img2 = os.path.join(trt_save_dir, img_name)
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        img = np.hstack((img1, img2))
        cv2.imwrite(os.path.join(compare_dir, img_name), img)
    os.system(f'rm -rf {torch_save_dir}')
    os.system(f'rm -rf {trt_save_dir}')