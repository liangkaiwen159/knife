_base_ = ['./icann-dino_w_encoder_with_loss_latter_wo_neck.py']
base_lr = 0.0001
model = dict(use_rpn=False,
             encoder_reasign=True,
             dn_cfg=dict(rpn_noise_flag=True, group_cfg=dict(dynamic=True, num_groups=1, num_dn_queries=100)))
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
        'rpn_head': dict(lr_mult=1),
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
