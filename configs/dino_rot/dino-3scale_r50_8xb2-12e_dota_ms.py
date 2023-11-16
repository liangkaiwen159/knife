_base_ = ['../_base_/datasets/dota_ms.py', '../_base_/default_runtime.py']
angle_version = 'le135'
nums_of_level = 4
model = dict(
    type='Rot_DINO',
    num_queries=2500,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    num_classes=15,
    latter=True,
    use_rpn=False,
    encoder_reasign=False,
    data_preprocessor=dict(type='DetDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=32,
                           boxtype2tensor=False),
    backbone=dict(type='ResNet',
                  depth=101,
                  num_stages=4,
                  out_indices=(1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=dict(type='BN', requires_grad=False),
                  norm_eval=True,
                  style='pytorch',
                  init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(type='ChannelMapper',
              in_channels=[512, 1024, 2048],
              kernel_size=1,
              out_channels=256,
              act_cfg=None,
              norm_cfg=dict(type='GN', num_groups=32),
              num_outs=nums_of_level,
              return_tuple=True),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=nums_of_level, dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        angle_version = angle_version,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=nums_of_level, dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='Rot_DINOHead',
        angle_version=angle_version,
        num_classes=15,
        sync_cls_avg_factor=True,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=5.0),  # 2.0 in DeformDETR
        # loss_bbox=dict(type='RL1Loss', loss_weight=5.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=3, num_dn_queries=1200)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(assigner=dict(
        type='HungarianAssigner',
        match_costs=[
            dict(type='FocalLossCost', weight=5.0),
            #  dict(type='RBBoxL1Cost', weight=5.0, box_format='cxcywhr'),
            dict(type='RBBoxL1Cost', weight=5.0, box_format='cxcywhr', use_normal_l1=True),
            dict(type='RotatedIoUCost', weight=2.0)
        ])),
    test_cfg=dict(max_per_img=600))  # 100 for DeformDETR

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(
        lr_mult=0.1)}))  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 36
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[30], gamma=0.1)]
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=0.0001 * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]


train_dataloader = dict(
    batch_size=1)
visualizer = dict(type='RotLocalVisualizer')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
randomness = dict(seed=42)
