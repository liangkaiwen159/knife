# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.DeformableDETRHead.predict_by_feat')
@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.DETRHead.predict_by_feat')
@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.Rot_DINOHead.predict_by_feat')

def detrhead__predict_by_feat__default(self,
                                       all_cls_scores_list: List[Tensor],
                                       all_bbox_preds_list: List[Tensor],
                                       all_layers_outputs_rots=None,
                                       batch_img_metas: List[dict]=None,
                                       rescale: bool = True):
    """Rewrite `predict_by_feat` of `FoveaHead` for default backend."""
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

    cls_scores = all_cls_scores_list[-1]
    bbox_preds = all_bbox_preds_list[-1]
    if all_layers_outputs_rots is not None:
        rot_preds = all_layers_outputs_rots[-1]
    img_shape = batch_img_metas[0]['img_shape']
    if isinstance(img_shape, list):
        img_shape = torch.tensor(
            img_shape, dtype=torch.long, device=cls_scores.device)
    img_shape = img_shape.unsqueeze(0)

    max_per_img = self.test_cfg.get('max_per_img', len(cls_scores[0]))
    batch_size = cls_scores.size(0)
    # `batch_index_offset` is used for the gather of concatenated tensor

    # supports dynamical batch inference
    if self.loss_cls.use_sigmoid:
        batch_index_offset = torch.arange(batch_size).to(
            cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(
            batch_size, max_per_img)
        cls_scores = cls_scores.sigmoid()
        scores, indexes = cls_scores.flatten(1).topk(max_per_img, dim=1)
        det_labels = indexes % self.num_classes
        bbox_index = indexes // self.num_classes
        bbox_index = (bbox_index + batch_index_offset).view(-1)
        bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
        bbox_preds = bbox_preds.view(batch_size, -1, 4)
        if all_layers_outputs_rots is not None:
            rot_preds = rot_preds.view(-1, 1)[bbox_index]
            rot_preds = rot_preds.view(batch_size, -1, 1)
    else:
        scores, det_labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
        scores, bbox_index = scores.topk(max_per_img, dim=1)
        batch_inds = torch.arange(
            batch_size, device=scores.device).unsqueeze(-1)
        bbox_preds = bbox_preds[batch_inds, bbox_index, ...]
        if all_layers_outputs_rots is not None:
            rot_preds = rot_preds[batch_inds, bbox_index, ...]
        # add unsqueeze to support tensorrt
        det_labels = det_labels.unsqueeze(-1)[batch_inds, bbox_index,
                                              ...].squeeze(-1)

    if all_layers_outputs_rots is None:
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
    det_bboxes = bbox_preds
    det_bboxes.clamp_(min=0., max=1.)
    shape_scale = img_shape.flip(1).repeat(1, 2).unsqueeze(1)
    det_bboxes = det_bboxes * shape_scale
    if all_layers_outputs_rots is not None:
        det_bboxes = torch.cat((det_bboxes, rot_preds), -1)
        det_bboxes = resize_rot_bbox_tensor(det_bboxes)
    det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)

    return det_bboxes, det_labels

def resize_rot_bbox_tensor(boxes):
    """Rescale boxes w.r.t. rescale_factor in-place.

    Note:
        Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
        w.r.t ``scale_facotr``. The difference is that ``resize_`` only
        changes the width and the height of boxes, but ``rescale_`` also
        rescales the box centers simultaneously.

    Args:
        scale_factor (Tuple[float, float]): factors for scaling boxes.
            The length should be 2.
    """
    ctrs, w, h, t = torch.split(boxes, [2, 1, 1, 1], dim=-1)
    cos_value, sin_value = torch.cos(t), torch.sin(t)

    # rescale width and height
    w = w * torch.sqrt((cos_value)**2 + (sin_value)**2)
    h = h * torch.sqrt((sin_value)**2 + (cos_value)**2)
    # recalculate theta
    t = torch.atan2(sin_value, cos_value)
    boxes = torch.cat([ctrs, w, h, t], dim=-1)

    return boxes