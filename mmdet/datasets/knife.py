# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Union
import numpy as np
from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset
import copy

@DATASETS.register_module()
class KnifeDataset(CocoDataset):
    """Dataset for COCO."""

    def __init__(self, *args, split: str = 'train', **kwargs) -> None:
        self.split = split
        super().__init__(*args, **kwargs)


    METAINFO = {
        'classes': ('knife'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60)]
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        nums_img = len(data_list)
        all_indices = np.arange(nums_img)
        np.random.seed(0)
        train_indices = sorted(np.random.choice(
            all_indices, size=int(4 / 5 * nums_img), replace=False))
        val_indices = sorted(np.setdiff1d(all_indices, train_indices))


        if self.split == 'train':
            data_list = np.array(data_list)[train_indices].tolist()
        elif self.split == 'val':
            data_list = np.array(data_list)[val_indices].tolist()

        return data_list
