_base_ = ['./icann-dino_w_encoder_with_loss_latter_wo_neck.py']
base_lr = 0.0001
model = dict(encoder_reasign=False,
             use_rpn=True,
             rpn_head=dict(_delete_=True,
                           type='YOLOv8Head',
                           head_module=dict(type='YOLOv8HeadModule',
                                            num_classes=80,
                                            in_channels=[256, 256, 256],
                                            widen_factor=1,
                                            reg_max=16,
                                            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                                            act_cfg=dict(type='SiLU', inplace=True),
                                            featmap_strides=[8, 16, 32]),
                           prior_generator=dict(type='MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
                           bbox_coder=dict(type='YOLODistancePointBBoxCoder'),
                           loss_cls=dict(type='CrossEntropyLoss',
                                         use_sigmoid=True,
                                         reduction='none',
                                         loss_weight=0.5),
                           loss_bbox=dict(type='IoULossYolo',
                                          iou_mode='ciou',
                                          bbox_format='xyxy',
                                          reduction='sum',
                                          loss_weight=7.5,
                                          return_iou=False),
                           loss_dfl=dict(type='DistributionFocalLoss', reduction='mean', loss_weight=0.375)),
             dn_cfg=dict(rpn_noise_flag=True, group_cfg=dict(dynamic=True, num_groups=1, num_dn_queries=100)))

train_dataloader = dict(
    batch_size=8,
    num_workers=2)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(_delete_=True,
                       custom_keys={
        'backbone': dict(lr_mult=0.1),
    }))  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[11], gamma=0.1)]
# param_scheduler = [
#     dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
#     dict(
#         # use cosine lr from 150 to 300 epoch
#         type='CosineAnnealingLR',
#         eta_min=base_lr * 0.1,
#         begin=max_epochs // 2,
#         end=max_epochs,
#         T_max=max_epochs // 2,
#         by_epoch=True,
#         convert_to_iter_based=True),
# ]
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
randomness = dict(seed=42)

# find_unused_parameters = True
