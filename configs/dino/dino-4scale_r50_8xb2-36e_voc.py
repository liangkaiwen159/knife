_base_ = './dino-4scale_r50_8xb2-12e_voc.py'
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]
train_dataloader = dict(batch_size=6,num_workers=6)
model=dict(num_queries=900)