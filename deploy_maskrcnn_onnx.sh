python tools/deploy.py\
    configs/mmdet/instance-seg/instance-seg_onnxruntime_dynamic.py\
    ../icann_dino_detr/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_knife.py\
    ../icann_dino_detr/work_dirs/mask-rcnn_r50_fpn_1x_knife/best_coco_bbox_mAP_epoch_500.pth\
    demo/knife.png\
    --work-dir mmdeploy_models/mmdet/ort\
    --device cpu\
    --dump-info