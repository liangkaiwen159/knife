_base_ = ['../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py']
nums_of_fpn = 3
num_classes = 80
num_queries = 900
img_scale = (640, 640)
# -----model related-----
strides = [8, 16, 32]  # Strides of multi-scale prior box
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
anchors = [
    [(12, 16), (19, 36), (40, 28)],  # P3/8
    [(36, 75), (76, 55), (72, 146)],  # P4/16
    [(142, 110), (192, 243), (459, 401)]  # P5/32
]
base_lr = 0.0001
loss_cls_weight = 3
loss_bbox_weight = 0.5
loss_obj_weight = 7
simota_candidate_topk = 10
simota_iou_weight = 3.0
simota_cls_weight = 1.0

model = dict(
    type='ICANN_DINO',
    num_queries=num_queries,  # num_matching_queries
    num_classes=num_classes,
    with_box_refine=True,
    as_two_stage=True,
    use_rpn=True,
    # use_rpn=False,
    data_preprocessor=dict(type='DetDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=32),
    backbone=dict(type='ResNet',
                  depth=50,
                  num_stages=4,
                  out_indices=(1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=norm_cfg,
                  norm_eval=True,
                  style='pytorch',
                  init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(type='ELANBlock', middle_ratio=0.5, block_ratio=0.25, num_blocks=4, num_convs_in_block=1),
        upsample_feats_cat_first=False,
        in_channels=[512, 1024, 2048],
        # The real output channel will be multiplied by 2
        out_channels=[128, 256, 512],
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        channelmapper_cfg=dict(type='ChannelMapper',
                               in_channels=[256, 512, 1024],
                               kernel_size=1,
                               out_channels=256,
                               act_cfg=None,
                               norm_cfg=dict(type='GN', num_groups=32),
                               num_outs=3)),
    # neck=dict(
    #     type='ChannelMapper',
    #     in_channels=[512, 1024, 2048],
    #     kernel_size=1,
    #     out_channels=256,
    #     act_cfg=None,
    #     norm_cfg=dict(type='GN', num_groups=32),
    #     num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=nums_of_fpn, dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    rpn_head=dict(
        type='ICANN_Rpn_Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            # type='ICANN_HeadModule',
            num_classes=num_classes,
            # in_channels=[256, 256, 256],
            in_channels=[256, 512, 1024],
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(type='mmdet.YOLOAnchorGenerator', base_sizes=anchors, strides=strides),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULossYolo',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=loss_bbox_weight,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight),
        # BatchYOLOv7Assigner params
        near_neighbor_thr=0.5,
        nums_proposals=num_queries,
        simota_candidate_topk=simota_candidate_topk,
        simota_iou_weight=simota_iou_weight,
        simota_cls_weight=simota_cls_weight,
        test_cfg=dict(
            # The config of multi-label for multi-class prediction.
            multi_label=True,
            # The number of boxes before NMS.
            nms_pre=30000,
            score_thr=0.0001,  # Threshold to filter out boxes.
            nms=dict(type='nms', iou_threshold=0.8),  # NMS type and threshold
            max_per_img=300)),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=nums_of_fpn, dropout=0.0),  # 0.1 for DeformDETR
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
        type='DINOHead',
        anchor_generator=dict(type='YOLOAnchorGenerator', base_sizes=anchors, strides=strides),
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(assigner=dict(type='HungarianAssigner',
                                 match_costs=[
                                     dict(type='FocalLossCost', weight=2.0),
                                     dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                                     dict(type='IoUCost', iou_mode='giou', weight=2.0)
                                 ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[
#             [
#                 dict(type='RandomChoiceResize',
#                      scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333),
#                              (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
#                      keep_ratio=True)
#             ],
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     # The radio of all image in train dataset < 7
#                     # follow the original implement
#                     scales=[(400, 4200), (500, 4200), (600, 4200)],
#                     keep_ratio=True),
#                 dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
#                 dict(type='RandomChoiceResize',
#                      scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333),
#                              (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
#                      keep_ratio=True)
#             ]
#         ]),
#     dict(type='PackDetInputs')
# ]
train_pipeline = [
    dict(
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', keep_ratio=True, scale=img_scale),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=6,
    sampler=dict(shuffle=True),
    # dataset=dict(indices=8000, filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))
    dataset=dict(filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

test_pipeline = [
    dict(
        type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
# val_dataloader = dict(
#     dataset=dict(indices=500, filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        'rpn_head': dict(lr_mult=0.1),
    }))  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[11], gamma=0.1)]
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.1,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
randomness = dict(seed=42)

# find_unused_parameters = True