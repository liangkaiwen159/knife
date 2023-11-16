# dataset settings
dataset_type = 'DOTADataset'
data_root = 'data/split_ss_dota/'
# scale = (640, 640)
scale = (1024, 1024)
backend_args = None
TRAIN_VAL = False

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='mmdet.RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', mask_border_value=(114, 114, 114), angle_range=180, prob=0.5, rotate_type='MMRotate'),
    dict(type='mmdet.Resize', scale=scale, keep_ratio=True),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=scale, keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=scale, keep_ratio=True),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_dataloader = dict(batch_size=2,
                        num_workers=2,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        batch_sampler=None,
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     ann_file='trainval/annfiles/' if TRAIN_VAL else 'train/annfiles/',
                                     data_prefix=dict(img_path='trainval/images/' if TRAIN_VAL else 'train/images/'),
                                     filter_cfg=dict(filter_empty_gt=False),
                                     pipeline=train_pipeline))
val_dataloader = dict(batch_size=1,
                      num_workers=2,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='trainval/annfiles/' if TRAIN_VAL else 'val/annfiles/',
                                   data_prefix=dict(img_path='trainval/images/' if TRAIN_VAL else 'val/images/'),
                                   test_mode=True,
                                   pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(batch_size=1,
                       num_workers=2,
                       persistent_workers=True,
                       drop_last=False,
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       dataset=dict(type=dataset_type,
                                    data_root=data_root,
                                    data_prefix=dict(img_path='test/images/'),
                                    test_mode=True,
                                    pipeline=test_pipeline))
test_evaluator = dict(type='DOTAMetric', format_only=True, merge_patches=True, outfile_prefix='./work_dirs/dota/Task1')
