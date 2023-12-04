python tools/deploy.py\
    configs/mmdet/detection/detection_tensorrt_static-1024x1024.py\
    ../icann_dino_detr/configs/dino_rot/dino-4scale_dota.py\
    ../icann_dino_detr/work_dirs/dino-4scale_dota/best_dota_mAP_0.73.pth\
    demo/P1384.png\
    --work-dir mmdeploy_models/dino/ort --device cuda --dump-info
