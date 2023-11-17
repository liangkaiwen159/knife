_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py', '../_base_/datasets/knife_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

num_classes = 1

default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(type='CheckpointHook', interval=50, save_best='auto'),
)

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=num_classes), mask_head=dict(num_classes=num_classes)))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000, val_interval=50)


# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[800, 900], gamma=0.1)
]
