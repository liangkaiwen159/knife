_base_ = ['./icann-dino-4scale-960_w_encoder_with_loss.py']
train_dataloader = dict(batch_size=4,
    num_workers=6)
# find_unused_parameters = True
max_epochs = 36
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[30], gamma=0.1)]
