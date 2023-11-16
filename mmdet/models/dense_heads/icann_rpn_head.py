# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from mmengine.logging import MMLogger, print_log
import logging
from mmcv.cnn import ConvModule
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh, scale_boxes)
from mmcv.ops import batched_nms
from ..utils import gt_instances_preprocess
from ..layers.transformer import inverse_sigmoid
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList, OptMultiConfig, InstanceList)
from mmengine.dist import get_dist_info
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from mmengine.config import ConfigDict
from torch import Tensor
from mmdet.structures.bbox.transforms import bbox_xyxy_to_cxcywh

from mmdet.registry import MODELS, TASK_UTILS
from ..layers import ImplicitA, ImplicitM
from ..task_modules.assigners.batch_yolov7_assigner import BatchYOLOv7Assigner
from ..task_modules.assigners.hungarian_assigner import HungarianAssigner
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule
from ..utils import unpack_gt_instances, get_parallel
import time
from joblib import delayed
from torch.nn.init import xavier_uniform_
import copy
import numpy as np


def make_divisible(x: float, widen_factor: float = 1.0, divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


@MODELS.register_module()
class YOLOv8HeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds, self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(5 / self.num_classes / (640 / stride)**2)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels, out_channels=4 * self.reg_max, kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels, out_channels=self.num_classes, kernel_size=1)))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList, reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class YOLOv7HeadModule(YOLOv5HeadModule):
    """YOLOv7Head head module used in YOLOv7."""

    def _init_layers(self):
        """initialize conv layers in YOLOv7 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Sequential(
                ImplicitA(self.in_channels[i]),
                nn.Conv2d(self.in_channels[i], self.num_base_priors * self.num_out_attrib, 1),
                ImplicitM(self.num_base_priors * self.num_out_attrib),
            )
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv7 head."""
        super(YOLOv5HeadModule, self).init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            mi = mi[1]  # nn.Conv2d

            b = mi.bias.data.view(3, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s)**2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))

            mi.bias.data = b.view(-1)


@MODELS.register_module()
class ICANN_HeadModule(YOLOv5HeadModule):
    """不使用anchor,每个特征点纯预测"""

    def _init_layers(self):
        """initialize conv layers in YOLOv7 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Sequential(
                ImplicitA(self.in_channels[i]),
                nn.Conv2d(self.in_channels[i], self.num_out_attrib, 1),
                ImplicitM(self.num_out_attrib),
            )
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv7 head."""
        super(YOLOv5HeadModule, self).init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            mi = mi[1]  # nn.Conv2d

            nn.init.xavier_uniform_(mi.weight)

    def forward_single(self, x: Tensor, convs: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, 1, self.num_out_attrib, ny, nx)

        cls_score = pred_map[:, :, 5:, ...].reshape(bs, -1, ny, nx)
        bbox_pred = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
        objectness = pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)

        return cls_score, bbox_pred, objectness


@MODELS.register_module()
class YOLOv7p6HeadModule(YOLOv5HeadModule):
    """YOLOv7Head head module used in YOLOv7."""

    def __init__(self,
                 *args,
                 main_out_channels: Sequence[int] = [256, 512, 768, 1024],
                 aux_out_channels: Sequence[int] = [320, 640, 960, 1280],
                 use_aux: bool = True,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        self.main_out_channels = main_out_channels
        self.aux_out_channels = aux_out_channels
        self.use_aux = use_aux
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """initialize conv layers in YOLOv7 head."""
        self.main_convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Sequential(
                ConvModule(self.in_channels[i],
                           self.main_out_channels[i],
                           3,
                           padding=1,
                           norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg),
                ImplicitA(self.main_out_channels[i]),
                nn.Conv2d(self.main_out_channels[i], self.num_base_priors * self.num_out_attrib, 1),
                ImplicitM(self.num_base_priors * self.num_out_attrib),
            )
            self.main_convs_pred.append(conv_pred)

        if self.use_aux:
            self.aux_convs_pred = nn.ModuleList()
            for i in range(self.num_levels):
                aux_pred = nn.Sequential(
                    ConvModule(self.in_channels[i],
                               self.aux_out_channels[i],
                               3,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(self.aux_out_channels[i], self.num_base_priors * self.num_out_attrib, 1))
                self.aux_convs_pred.append(aux_pred)
        else:
            self.aux_convs_pred = [None] * len(self.main_convs_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        super(YOLOv5HeadModule, self).init_weights()
        for mi, aux, s in zip(self.main_convs_pred, self.aux_convs_pred, self.featmap_strides):  # from
            mi = mi[2]  # nn.Conv2d
            b = mi.bias.data.view(3, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s)**2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
            mi.bias.data = b.view(-1)

            if self.use_aux:
                aux = aux[1]  # nn.Conv2d
                b = aux.bias.data.view(3, -1)
                # obj (8 objects per 640 image)
                b.data[:, 4] += math.log(8 / (640 / s)**2)
                b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
                mi.bias.data = b.view(-1)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.main_convs_pred, self.aux_convs_pred)

    def forward_single(self, x: Tensor, convs: nn.Module,
                       aux_convs: Optional[nn.Module]) \
            -> Tuple[Union[Tensor, List], Union[Tensor, List],
                     Union[Tensor, List]]:
        """Forward feature of a single scale level."""

        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors, self.num_out_attrib, ny, nx)

        cls_score = pred_map[:, :, 5:, ...].reshape(bs, -1, ny, nx)
        bbox_pred = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
        objectness = pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)

        if not self.training or not self.use_aux:
            return cls_score, bbox_pred, objectness
        else:
            aux_pred_map = aux_convs(x)
            aux_pred_map = aux_pred_map.view(bs, self.num_base_priors, self.num_out_attrib, ny, nx)
            aux_cls_score = aux_pred_map[:, :, 5:, ...].reshape(bs, -1, ny, nx)
            aux_bbox_pred = aux_pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
            aux_objectness = aux_pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)

            return [cls_score, aux_cls_score], [bbox_pred, aux_bbox_pred], [objectness, aux_objectness]


@MODELS.register_module()
class ICANN_Rpn_Head(YOLOv5Head):
    """YOLOv7Head head used in `YOLOv7 <https://arxiv.org/abs/2207.02696>`_.

    Args:
        simota_candidate_topk (int): The candidate top-k which used to
            get top-k ious to calculate dynamic-k in BatchYOLOv7Assigner.
            Defaults to 10.
        simota_iou_weight (float): The scale factor for regression
            iou cost in BatchYOLOv7Assigner. Defaults to 3.0.
        simota_cls_weight (float): The scale factor for classification
            cost in BatchYOLOv7Assigner. Defaults to 1.0.
    """

    def __init__(
        self,
        head_module: ConfigType,
        prior_generator: ConfigType = dict(type='mmdet.YOLOAnchorGenerator',
                                           base_sizes=[[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)],
                                                       [(116, 90), (156, 198), (373, 326)]],
                                           strides=[8, 16, 32]),
        bbox_coder: ConfigType = dict(type='YOLOv5BBoxCoder'),
        loss_cls: ConfigType = dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.5),
        loss_bbox: ConfigType = dict(type='IoULoss',
                                     iou_mode='ciou',
                                     bbox_format='xywh',
                                     eps=1e-7,
                                     reduction='mean',
                                     loss_weight=0.05,
                                     return_iou=True),
        loss_obj: ConfigType = dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1.0),
        prior_match_thr: float = 4.0,
        near_neighbor_thr: float = 0.5,
        ignore_iof_thr: float = -1.0,
        obj_level_weights: List[float] = [4.0, 1.0, 0.4],
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        nums_proposals: int = 300,
        init_cfg: OptMultiConfig = None,
        simota_candidate_topk: int = 20,
        simota_iou_weight: float = 3.0,
        simota_cls_weight: float = 1.0,
        aux_loss_weights: float = 0.25,
        num_dn_queries: int = 100,
    ):
        super(YOLOv5Head, self).__init__(init_cfg=init_cfg)

        self.head_module = MODELS.build(head_module)
        self.num_classes = self.head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_dn_queries = num_dn_queries

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.nums_proposals = nums_proposals

        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.loss_cls: nn.Module = MODELS.build(loss_cls)
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)
        self.loss_obj: nn.Module = MODELS.build(loss_obj)

        self.featmap_sizes = [torch.empty(1)] * self.num_levels

        self.prior_match_thr = prior_match_thr
        self.near_neighbor_thr = near_neighbor_thr
        self.obj_level_weights = obj_level_weights
        self.ignore_iof_thr = ignore_iof_thr
        self.parallel = get_parallel(n_jobs=1)

        self.special_init()
        self.aux_loss_weights = aux_loss_weights
        self.assigner = BatchYOLOv7Assigner(num_classes=self.num_classes,
                                            num_base_priors=self.num_base_priors,
                                            featmap_strides=self.featmap_strides,
                                            prior_match_thr=self.prior_match_thr,
                                            candidate_topk=simota_candidate_topk,
                                            iou_weight=simota_iou_weight,
                                            cls_weight=simota_cls_weight)

        self.dn_assinger = HungarianAssigner(match_costs=[
            dict(type='FocalLossCost', weight=2.0),
            dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            dict(type='IoUCost', iou_mode='giou', weight=2.0)
        ])
        assert hasattr(self, 'label_embedding')
        
        # self.label_embedding = nn.Embedding(self.num_classes, 256)
        # xavier_uniform_(self.label_embedding.weight)

        self.logger = MMLogger.get_current_instance()
        self.logger_name = self.logger.instance_name

    def _predict(self, x: Tuple[Tensor], batch_data_samples: SampleList, rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        self.head_module.training = False
        outs = self(x)
        self.head_module.training = True

        if isinstance(batch_data_samples, list):
            batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
        else:
            batch_gt_instances = batch_data_samples['bboxes_labels']
            batch_img_metas = batch_data_samples['img_metas']
            batch_gt_instances_ignore = None

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale,
                                           batch_gt_instances=batch_gt_instances)
        return predictions

    def _predict_by_feat(self,
                         cls_scores: List[Tensor],
                         bbox_preds: List[Tensor],
                         objectnesses: Optional[List[Tensor]] = None,
                         batch_img_metas: Optional[List[dict]] = None,
                         cfg: Optional[ConfigDict] = None,
                         rescale: bool = True,
                         with_nms: bool = True,
                         batch_gt_instances=None) -> List[InstanceData]:

        gt_labels_list = []
        for gt_instance in batch_gt_instances:
            gt_labels_list.append(gt_instance.labels.detach())

        # for gen num groups
        num_target_list = [len(bboxes) for bboxes in gt_labels_list]
        max_num_target = max(num_target_list)

        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label
        cfg.max_per_img = max_num_target

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(featmap_sizes,
                                                                dtype=cls_scores[0].dtype,
                                                                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel() * self.num_base_priors, ), stride)
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        label_embed_list = []
        coord_list = []
        results_list = []
        for (bboxes, scores, objectness, img_meta, num_targets) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                                                                       flatten_objectness, batch_img_metas,
                                                                       num_target_list):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get('yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                label_embed_list.append(torch.zeros((max_num_target * 2, 256), device=bboxes.device))
                coord_list.append(torch.zeros((max_num_target * 2, 4), device=bboxes.device))
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(scores,
                                                                       score_thr,
                                                                       nms_pre,
                                                                       results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(scores, score_thr, nms_pre)

            results = InstanceData(scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor(
                        [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
                results.bboxes /= results.bboxes.new_tensor(scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta,
                                              num_targets=num_targets)
            # results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 0::2] = results.bboxes[:, 0::2] / ori_shape[1]
            # results.bboxes[:, 1::2].clamp_(0, ori_shape[0])
            results.bboxes[:, 1::2] = results.bboxes[:, 1::2] / ori_shape[0]

            results.bboxes = bbox_xyxy_to_cxcywh(results.bboxes)
            results.bboxes.clamp_(0.0, 1.0)

            results.bboxes = results.bboxes.detach()
            results.labels = results.labels.detach()

            if num_targets == 0:
                label_embed = torch.zeros((max_num_target * 2, 256), device=bboxes.device)
                coord = torch.zeros((max_num_target * 2, 4), device=bboxes.device)
                label_embed_list.append(label_embed)
                coord_list.append(coord)
            elif len(results) < max_num_target * 2:
                diff = max_num_target * 2 - len(results)
                label_embed = torch.cat(
                    (self.label_embedding(results.labels), torch.zeros((diff, 256), device=bboxes.device)))
                coord = torch.cat((results.bboxes, torch.zeros((diff, 4), device=bboxes.device)))
                label_embed_list.append(label_embed)
                coord_list.append(coord)
            else:
                label_embed_list.append(self.label_embedding(results.labels))
                coord_list.append(results.bboxes)

            results_list.append(results)

        return torch.stack(coord_list, 0), torch.stack(label_embed_list, 0), 1

    # def _bbox_post_process(self,
    #                        results: InstanceData,
    #                        cfg: ConfigDict,
    #                        rescale: bool = False,
    #                        with_nms: bool = True,
    #                        img_meta: Optional[dict] = None,
    #                        num_targets: int = 0) -> InstanceData:
    #     if rescale:
    #         assert img_meta.get('scale_factor') is not None
    #         scale_factor = [1 / s for s in img_meta['scale_factor']]
    #         results.bboxes = scale_boxes(results.bboxes, scale_factor)

    #     if hasattr(results, 'score_factors'):
    #         # TODO： Add sqrt operation in order to be consistent with
    #         #  the paper.
    #         score_factors = results.pop('score_factors')
    #         results.scores = results.scores * score_factors

    #     # filter small size bboxes
    #     if cfg.get('min_bbox_size', -1) >= 0:
    #         w, h = get_box_wh(results.bboxes)
    #         valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
    #         if not valid_mask.all():
    #             results = results[valid_mask]

    #     # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
    #     if with_nms and results.bboxes.numel() > 0:
    #         bboxes = get_box_tensor(results.bboxes)
    #         det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
    #                                             results.labels, cfg.nms)
    #         results = results[keep_idxs]
    #         # some nms would reweight the score, such as softnms
    #         results.scores = det_bboxes[:, -1]
    #         results_pos = results[:num_targets]
    #         result_neg = results[-num_targets:]
    #     else:
    #         empty_results = InstanceData()
    #         empty_results.bboxes = torch.zeros((num_targets * 2, 4),
    #                                            device=results.bboxes.device)
    #         empty_results.scores = torch.zeros((num_targets * 2, ),
    #                                            device=results.bboxes.device)
    #         empty_results.labels = torch.zeros(
    #             (num_targets * 2, ), device=results.bboxes.device).int()
    #         return empty_results
    #     results = InstanceData.cat([results_pos, result_neg])
    #     return results

    def loss_by_feat(self,
                     cls_scores: Sequence[Union[Tensor, List]],
                     bbox_preds: Sequence[Union[Tensor, List]],
                     objectnesses: Sequence[Union[Tensor, List]],
                     batch_gt_instances: Sequence[InstanceData],
                     batch_img_metas: Sequence[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """

        if isinstance(cls_scores[0], Sequence):
            with_aux = True
            batch_size = cls_scores[0][0].shape[0]
            device = cls_scores[0][0].device

            bbox_preds_main, bbox_preds_aux = zip(*bbox_preds)
            objectnesses_main, objectnesses_aux = zip(*objectnesses)
            cls_scores_main, cls_scores_aux = zip(*cls_scores)

            head_preds = self._merge_predict_results(bbox_preds_main, objectnesses_main, cls_scores_main)
            head_preds_aux = self._merge_predict_results(bbox_preds_aux, objectnesses_aux, cls_scores_aux)
        else:
            with_aux = False
            batch_size = cls_scores[0].shape[0]
            device = cls_scores[0].device

            head_preds = self._merge_predict_results(bbox_preds, objectnesses, cls_scores)

        # Convert gt to norm xywh format
        # (num_base_priors, num_batch_gt, 7)
        # 7 is mean (batch_idx, cls_id, x_norm, y_norm,
        # w_norm, h_norm, prior_idx)
        batch_targets_normed = self._convert_gt_to_norm_format(batch_gt_instances, batch_img_metas)

        scaled_factors = [torch.tensor(head_pred.shape, device=device)[[3, 2, 3, 2]] for head_pred in head_preds]

        loss_cls, loss_obj, loss_box = self._calc_loss(head_preds=head_preds,
                                                       head_preds_aux=None,
                                                       batch_targets_normed=batch_targets_normed,
                                                       near_neighbor_thr=self.near_neighbor_thr,
                                                       scaled_factors=scaled_factors,
                                                       batch_img_metas=batch_img_metas,
                                                       device=device)

        if with_aux:
            loss_cls_aux, loss_obj_aux, loss_box_aux = self._calc_loss(head_preds=head_preds,
                                                                       head_preds_aux=head_preds_aux,
                                                                       batch_targets_normed=batch_targets_normed,
                                                                       near_neighbor_thr=self.near_neighbor_thr * 2,
                                                                       scaled_factors=scaled_factors,
                                                                       batch_img_metas=batch_img_metas,
                                                                       device=device)
            loss_cls += self.aux_loss_weights * loss_cls_aux
            loss_obj += self.aux_loss_weights * loss_obj_aux
            loss_box += self.aux_loss_weights * loss_box_aux

        _, world_size = get_dist_info()
        return dict(rpn_loss_cls=loss_cls * batch_size * world_size,
                    rpn_loss_obj=loss_obj * batch_size * world_size,
                    rpn_loss_bbox=loss_box * batch_size * world_size)

    def _calc_loss(self, head_preds, head_preds_aux, batch_targets_normed, near_neighbor_thr, scaled_factors,
                   batch_img_metas, device):
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        assigner_results = self.assigner(head_preds,
                                         batch_targets_normed,
                                         batch_img_metas[0]['batch_input_shape'],
                                         self.priors_base_sizes,
                                         self.grid_offset,
                                         near_neighbor_thr=near_neighbor_thr)
        # mlvl is mean multi_level
        mlvl_positive_infos = assigner_results['mlvl_positive_infos']
        mlvl_priors = assigner_results['mlvl_priors']
        mlvl_targets_normed = assigner_results['mlvl_targets_normed']

        if head_preds_aux is not None:
            # This is mean calc aux branch loss
            head_preds = head_preds_aux

        for i, head_pred in enumerate(head_preds):
            batch_inds, proir_idx, grid_x, grid_y = mlvl_positive_infos[i].T
            num_pred_positive = batch_inds.shape[0]
            target_obj = torch.zeros_like(head_pred[..., 0])
            # empty positive sampler
            if num_pred_positive == 0:
                loss_box += head_pred[..., :4].sum() * 0
                loss_cls += head_pred[..., 5:].sum() * 0
                loss_obj += self.loss_obj(head_pred[..., 4], target_obj) * self.obj_level_weights[i]
                continue

            priors = mlvl_priors[i]
            targets_normed = mlvl_targets_normed[i]

            head_pred_positive = head_pred[batch_inds, proir_idx, grid_y, grid_x]

            # calc bbox loss
            grid_xy = torch.stack([grid_x, grid_y], dim=1)
            decoded_pred_bbox = self._decode_bbox_to_xywh(head_pred_positive[:, :4], priors, grid_xy)
            target_bbox_scaled = targets_normed[:, 2:6] * scaled_factors[i]

            loss_box_i, iou = self.loss_bbox(decoded_pred_bbox, target_bbox_scaled)
            loss_box += loss_box_i

            # calc obj loss
            target_obj[batch_inds, proir_idx, grid_y, grid_x] = iou.detach().clamp(0).type(target_obj.dtype)
            loss_obj += self.loss_obj(head_pred[..., 4], target_obj) * self.obj_level_weights[i]

            # calc cls loss
            if self.num_classes > 1:
                pred_cls_scores = targets_normed[:, 1].long()
                target_class = torch.full_like(head_pred_positive[:, 5:], 0., device=device)
                target_class[range(num_pred_positive), pred_cls_scores] = 1.
                loss_cls += self.loss_cls(head_pred_positive[:, 5:], target_class)
            else:
                loss_cls += head_pred_positive[:, 5:].sum() * 0
        return loss_cls, loss_obj, loss_box

    def _merge_predict_results(self, bbox_preds: Sequence[Tensor], objectnesses: Sequence[Tensor],
                               cls_scores: Sequence[Tensor]) -> List[Tensor]:
        """Merge predict output from 3 heads.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).

        Returns:
              List[Tensor]: Merged output.
        """
        head_preds = []
        for bbox_pred, objectness, cls_score in zip(bbox_preds, objectnesses, cls_scores):
            b, _, h, w = bbox_pred.shape
            bbox_pred = bbox_pred.reshape(b, self.num_base_priors, -1, h, w)
            objectness = objectness.reshape(b, self.num_base_priors, -1, h, w)
            cls_score = cls_score.reshape(b, self.num_base_priors, -1, h, w)
            head_pred = torch.cat([bbox_pred, objectness, cls_score], dim=2).permute(0, 1, 3, 4, 2).contiguous()
            head_preds.append(head_pred)
        return head_preds

    def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes, grid_xy) -> Tensor:
        bbox_pred = bbox_pred.sigmoid()
        pred_xy = bbox_pred[:, :2] * 2 - 0.5 + grid_xy
        pred_wh = (bbox_pred[:, 2:] * 2)**2 * priors_base_sizes
        decoded_bbox_pred = torch.cat((pred_xy, pred_wh), dim=-1)
        return decoded_bbox_pred

    # def forward(self, x, batch_data_samples: Union[list, dict]):
    #     output = self.head_module(x)
    #     if isinstance(batch_data_samples, list):
    #         batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
    #     else:
    #         batch_gt_instances = batch_data_samples['bboxes_labels']
    #         batch_img_metas = batch_data_samples['img_metas']
    #         batch_gt_instances_ignore = None
    #     # batch_gt_instances_copy = copy.deepcopy(batch_gt_instances)
    #     # for i in range(len(batch_gt_instances_copy)):
    #     #     batch_gt_instances_copy[i].labels.fill_(0)

    #     # results = self.gen_cls_obj_corrd(*output, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
    #     results = self.gen_pos_neg_as_noise(*output, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
    #     return results

    def gen_pos_neg_query(
            self,
            cls_scores: Sequence[Union[Tensor, List]],  # 3个头
            bbox_preds: Sequence[Union[Tensor, List]],
            objectnesses: Sequence[Union[Tensor, List]],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None):
        t1 = time.time()

        if isinstance(cls_scores[0], Sequence):
            with_aux = True
            batch_size = cls_scores[0][0].shape[0]
            device = cls_scores[0][0].device

            bbox_preds_main, bbox_preds_aux = zip(*bbox_preds)
            objectnesses_main, objectnesses_aux = zip(*objectnesses)
            cls_scores_main, cls_scores_aux = zip(*cls_scores)

            head_preds = self._merge_predict_results(bbox_preds_main, objectnesses_main, cls_scores_main)
            head_preds_aux = self._merge_predict_results(bbox_preds_aux, objectnesses_aux, cls_scores_aux)
        else:
            with_aux = False
            batch_size = cls_scores[0].shape[0]
            device = cls_scores[0].device

            head_preds = self._merge_predict_results(bbox_preds, objectnesses, cls_scores)

        # Convert gt to norm xywh format
        # (num_base_priors, num_batch_gt, 7)
        # 7 is mean (batch_idx, cls_id, x_norm, y_norm,
        # w_norm, h_norm, prior_idx)
        batch_targets_normed = self._convert_gt_to_norm_format(batch_gt_instances, batch_img_metas)

        scaled_factors = [torch.tensor(head_pred.shape, device=device)[[3, 2, 3, 2]] for head_pred in head_preds]

        assigner_results = self.assigner(head_preds,
                                         batch_targets_normed,
                                         batch_img_metas[0]['batch_input_shape'],
                                         self.priors_base_sizes,
                                         self.grid_offset,
                                         near_neighbor_thr=self.near_neighbor_thr)
        # mlvl is mean multi_level
        mlvl_positive_infos = assigner_results['mlvl_positive_infos']
        mlvl_priors = assigner_results['mlvl_priors']
        mlvl_targets_normed = assigner_results['mlvl_targets_normed']

        positive_decoded_pred_bboxes_normed = [[] for _ in range(batch_size)]
        negative_decoded_pred_bboxes_normed = [[] for _ in range(batch_size)]
        positive_class_info = [[] for _ in range(batch_size)]
        negative_class_info = [[] for _ in range(batch_size)]

        # for i, head_pred in enumerate(head_preds):
        #     for batch_idx in range(batch_size):
        def _parallel_task(batch_idx, head_pred):
            batch_inds, proir_idx, grid_x, grid_y = mlvl_positive_infos[i][mlvl_positive_infos[i][:, 0] == batch_idx].T
            num_pred_positive = batch_inds.shape[0]
            # empty positive sampler
            if num_pred_positive > 0:
                priors = mlvl_priors[i][mlvl_positive_infos[i][:, 0] == batch_idx]

                head_pred_positive = head_pred[batch_inds, proir_idx, grid_y, grid_x]
                grid_xy = torch.stack([grid_x, grid_y], dim=1)
                decoded_pred_bbox = self._decode_bbox_to_xywh(head_pred_positive[:, :4], priors, grid_xy)
                decoded_pred_bbox_normed = decoded_pred_bbox / scaled_factors[i]
                positive_class_info[batch_idx].append(head_pred_positive[:, 4:])
                positive_decoded_pred_bboxes_normed[batch_idx].append(decoded_pred_bbox_normed)
            else:
                positive_decoded_pred_bboxes_normed[batch_idx].append(torch.empty((0, 4), device=device))
                positive_class_info[batch_idx].append(torch.empty((0, 81), device=device))

            # negative sampler
            # if math.ceil(self.nums_proposals / 3) > num_pred_positive:
            head_pred_negative = head_pred.clone()[batch_idx]
            if proir_idx.numel():
                head_pred_negative[proir_idx, grid_y, grid_x] = float('inf')
            batch_flatten_map = torch.zeros(head_pred_negative.flatten(0, 2).shape[0], dtype=bool)
            head_pred_negative_topk_idx = torch.topk(head_pred_negative.flatten(0, 2)[..., 4],
                                                     math.ceil(self.nums_proposals / 3),
                                                     largest=False)[1]
            batch_flatten_map[head_pred_negative_topk_idx] = True
            batch_topk_map = batch_flatten_map.reshape(*head_pred_negative.shape[:-1])
            neg_prios_idx, neg_grid_y, neg_grid_x = batch_topk_map.nonzero().T
            head_pred_negative_topk = head_pred_negative[neg_prios_idx, neg_grid_y, neg_grid_x]
            neg_grid_xy = torch.stack([neg_grid_x, neg_grid_y], dim=1).to(device)
            neg_piros = self.priors_base_sizes[i][neg_prios_idx]
            neg_decoded_pred_bbox = self._decode_bbox_to_xywh(
                head_pred_negative_topk[:, :4],
                neg_piros,
                neg_grid_xy,
            )
            neg_decoded_pred_bbox_normed = neg_decoded_pred_bbox / scaled_factors[i]
            negative_class_info[batch_idx].append(head_pred_negative_topk[:, 4:])
            negative_decoded_pred_bboxes_normed[batch_idx].append(neg_decoded_pred_bbox_normed)
            # else:
            #     negative_decoded_pred_bboxes_normed[batch_idx].append(torch.empty((0, 4), device=device))
            #     negative_class_info[batch_idx].append(torch.empty((0, 80), device=device))

        for i, head_pred in enumerate(head_preds):
            task1 = [delayed(_parallel_task)(batch_idx, head_pred) for batch_idx in range(batch_size)]
            self.parallel(task1)

        rpn_positive_info = []
        rpn_coord_info = []
        rpn_class_info = []
        t2 = time.time()

        def _parallel_task2(batch_idx):
            # for batch_idx in range(batch_size):
            positive = torch.cat(positive_decoded_pred_bboxes_normed[batch_idx], 0)
            negative = torch.cat(negative_decoded_pred_bboxes_normed[batch_idx], 0)
            positive_class = torch.cat(positive_class_info[batch_idx], 0)
            negative_class = torch.cat(negative_class_info[batch_idx], 0)
            if positive.shape[0] < self.nums_proposals // 2:
                diff = self.nums_proposals - positive.shape[0]
                negative = negative[:diff]
                negative_class = negative_class[:diff, 1:]
                positive_class = positive_class[:, 1:]
            else:
                pos_topk_index = torch.topk(positive_class[:, 0], self.nums_proposals // 2)[1]
                positive = positive[pos_topk_index]
                positive_class = positive_class[pos_topk_index][:, 1:]

                negative = negative[:self.nums_proposals // 2]
                negative_class = negative_class[:self.nums_proposals // 2, 1:]

            all_coord_info = torch.cat((positive, negative), 0)
            all_class_info = torch.cat((positive_class, negative_class), 0)
            rpn_positive_info.append(positive.shape[0])
            rpn_coord_info.append(all_coord_info)
            rpn_class_info.append(all_class_info)

        task2 = [delayed(_parallel_task2)(batch_idx) for batch_idx in range(batch_size)]
        self.parallel(task2)
        # t3 = time.time()
        # print(f'process time1 {t2-t1}s')
        # print(f'process time2 {t3-t1}s')
        return torch.stack(rpn_coord_info, 0).clamp(0.0, 0.9999), torch.stack(rpn_class_info, 0), rpn_positive_info

    def gen_cls_obj_corrd(
            self,
            cls_scores: Sequence[Union[Tensor, List]],  # 3个头
            bbox_preds: Sequence[Union[Tensor, List]],
            objectnesses: Sequence[Union[Tensor, List]],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None):

        cls_list = []
        obj_list = []
        coord_list = []

        for cls, obj, coord in zip(cls_scores, objectnesses, bbox_preds):
            cls_list.append(cls.flatten(2, 3).permute(0, 2, 1).contiguous())
            obj_list.append(obj.flatten(2, 3).permute(0, 2, 1).contiguous())
            coord_list.append(coord.flatten(2, 3).permute(0, 2, 1).contiguous())

        return torch.cat(cls_list, 1), torch.cat(obj_list, 1), torch.cat(coord_list, 1)

    def gen_pos_neg_as_noise(
            self,
            cls_scores: Sequence[Union[Tensor, List]],  # 3个头
            bbox_preds: Sequence[Union[Tensor, List]],
            objectnesses: Sequence[Union[Tensor, List]],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None):
        t1 = time.time()

        batch_size = cls_scores[0].shape[0]
        device = cls_scores[0].device

        gt_labels_list = []
        for gt_instance in batch_gt_instances:
            gt_labels_list.append(gt_instance.labels.detach())

        # for gen num groups
        num_target_list = [len(bboxes) for bboxes in gt_labels_list]
        max_num_target = max(num_target_list)
        if max_num_target == 0:
            num_groups = 1
        else:
            num_groups = self.num_dn_queries // max_num_target

        if num_groups < 1:
            num_groups = 1

        num_rpn_noise = num_groups * max_num_target

        head_preds = self._merge_predict_results(bbox_preds, objectnesses, cls_scores)

        # Convert gt to norm xywh format
        # (num_base_priors, num_batch_gt, 7)
        # 7 is mean (batch_idx, cls_id, x_norm, y_norm,
        # w_norm, h_norm, prior_idx)
        batch_targets_normed = self._convert_gt_to_norm_format(batch_gt_instances, batch_img_metas)

        scaled_factors = [torch.tensor(head_pred.shape, device=device)[[3, 2, 3, 2]] for head_pred in head_preds]

        assigner_results = self.assigner(head_preds,
                                         batch_targets_normed,
                                         batch_img_metas[0]['batch_input_shape'],
                                         self.priors_base_sizes,
                                         self.grid_offset,
                                         near_neighbor_thr=self.near_neighbor_thr)
        # mlvl is mean multi_level
        mlvl_positive_infos = assigner_results['mlvl_positive_infos']
        mlvl_priors = assigner_results['mlvl_priors']
        mlvl_targets_normed = assigner_results['mlvl_targets_normed']

        positive_decoded_pred_bboxes_normed = [[] for _ in range(batch_size)]
        negative_decoded_pred_bboxes_normed = [[] for _ in range(batch_size)]
        positive_class_info = [[] for _ in range(batch_size)]
        negative_class_info = [[] for _ in range(batch_size)]

        for i, head_pred in enumerate(head_preds):
            for batch_idx in range(batch_size):
                if mlvl_positive_infos[i].numel() and max_num_target != 0:
                    cur_batch_pos_idx = mlvl_positive_infos[i][:, 0] == batch_idx
                    num_pred_positive = mlvl_positive_infos[i][cur_batch_pos_idx].T[0].shape[0]
                    batch_inds, proir_idx, grid_x, grid_y = mlvl_positive_infos[i][cur_batch_pos_idx].T
                    if num_pred_positive > 0:
                        priors = mlvl_priors[i][cur_batch_pos_idx]
                        head_pred_positive = head_pred[batch_inds, proir_idx, grid_y, grid_x]
                        grid_xy = torch.stack([grid_x, grid_y], dim=1)
                        decoded_pred_bbox = self._decode_bbox_to_xywh(head_pred_positive[:, :4], priors, grid_xy)
                        decoded_pred_bbox_normed = decoded_pred_bbox / scaled_factors[i]
                        positive_class_info[batch_idx].append(head_pred_positive[:, 4:])
                        positive_decoded_pred_bboxes_normed[batch_idx].append(decoded_pred_bbox_normed)
                    else:
                        positive_decoded_pred_bboxes_normed[batch_idx].append(torch.empty((0, 4), device=device))
                        positive_class_info[batch_idx].append(torch.empty((0, 81), device=device))
                        proir_idx, grid_x, grid_y = torch.empty((3, 0))
                    # negative sampler
                    head_pred_negative = head_pred[batch_idx].clone()
                    if proir_idx.numel():
                        head_pred_negative[proir_idx, grid_y, grid_x] = float('inf')
                    batch_flatten_map = torch.zeros(head_pred_negative.flatten(0, 2).shape[0], dtype=bool)
                    head_pred_negative_topk_idx = torch.topk(head_pred_negative.flatten(0, 2)[..., 4],
                                                             num_rpn_noise,
                                                             largest=False)[1]
                    batch_flatten_map[head_pred_negative_topk_idx] = True
                    batch_topk_map = batch_flatten_map.reshape(*head_pred_negative.shape[:-1])
                    neg_prios_idx, neg_grid_y, neg_grid_x = batch_topk_map.nonzero().T
                    head_pred_negative_topk = head_pred_negative[neg_prios_idx, neg_grid_y, neg_grid_x]
                    neg_grid_xy = torch.stack([neg_grid_x, neg_grid_y], dim=1).to(device)
                    neg_piros = self.priors_base_sizes[i][neg_prios_idx]
                    neg_decoded_pred_bbox = self._decode_bbox_to_xywh(
                        head_pred_negative_topk[:, :4],
                        neg_piros,
                        neg_grid_xy,
                    )
                    neg_decoded_pred_bbox_normed = neg_decoded_pred_bbox / scaled_factors[i]
                    negative_class_info[batch_idx].append(head_pred_negative_topk[:, 4:])
                    negative_decoded_pred_bboxes_normed[batch_idx].append(neg_decoded_pred_bbox_normed)
                else:
                    positive_decoded_pred_bboxes_normed[batch_idx].append(torch.empty((0, 4), device=device))
                    positive_class_info[batch_idx].append(torch.empty((0, 81), device=device))
                    negative_decoded_pred_bboxes_normed[batch_idx].append(torch.empty((0, 4), device=device))
                    negative_class_info[batch_idx].append(torch.empty((0, 81), device=device))
        rpn_positive_info = []
        rpn_coord_info = []
        rpn_label_embed_info = []
        t2 = time.time()

        for batch_idx in range(batch_size):
            positive = torch.cat(positive_decoded_pred_bboxes_normed[batch_idx], 0)
            negative = torch.cat(negative_decoded_pred_bboxes_normed[batch_idx], 0)
            positive_class = torch.cat(positive_class_info[batch_idx], 0)
            negative_class = torch.cat(negative_class_info[batch_idx], 0)

            if negative.shape[0] != 0:
                negative_topk_indices = torch.topk(negative_class[:, 0], num_rpn_noise, largest=False)[1]
                negative_class = negative_class[negative_topk_indices][:, 1:]
                negative_label_embed = self.label_embedding(negative_class.max(-1)[1])
                negative = negative[negative_topk_indices]
            else:
                negative_label_embed = torch.zeros((num_rpn_noise, 256)).to(device)
                negative = torch.zeros((num_rpn_noise, 4)).to(device)

            if positive.shape[0] == 0:
                positive_label_embed = torch.zeros((num_rpn_noise, 256)).to(device)
                positive = torch.zeros((num_rpn_noise, 4)).to(device)
            elif positive.shape[0] < num_rpn_noise:
                diff = num_rpn_noise - positive.shape[0]
                positive_label_embed = self.label_embedding(positive_class[:, 1:].max(-1)[1])
                positive_label_embed = torch.cat((positive_label_embed, torch.zeros((diff, 256)).to(device)), dim=0)
                positive = torch.cat((positive, torch.zeros((diff, 4)).to(device)), dim=0)

            else:
                pos_topk_index = torch.topk(positive_class[:, 0], max_num_target)[1]
                positive = positive[pos_topk_index]
                positive_class = positive_class[pos_topk_index][:, 1:]
                positive_label_embed = self.label_embedding(positive_class.max(-1)[1])

            all_coord_info = torch.cat((positive, negative), 0)
            all_class_info = torch.cat((positive_label_embed, negative_label_embed), 0)

            rpn_positive_info.append(positive.shape[0])
            rpn_coord_info.append(all_coord_info)
            rpn_label_embed_info.append(all_class_info)

        # t3 = time.time()
        # print(f'process time1 {t2-t1}s')
        # print(f'process time2 {t3-t1}s')
        return_coord_info = torch.stack(rpn_coord_info, 0).clamp(0.0, 1.0)
        return_label_embed_info = torch.stack(rpn_label_embed_info, 0)
        if not return_coord_info.numel():
            num_groups = 0
        return return_coord_info, return_label_embed_info, num_groups

    def gen_noise(self, x: Tuple[Tensor], batch_data_samples: SampleList, rescale: bool = False):
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        self.head_module.training = False
        outs = self(x)
        self.head_module.training = True

        if isinstance(batch_data_samples, list):
            batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
        else:
            batch_gt_instances = batch_data_samples['bboxes_labels']
            batch_img_metas = batch_data_samples['img_metas']
            batch_gt_instances_ignore = None

        predictions = self.gen_noise_by_hungarian(*outs,
                                                  batch_img_metas=batch_img_metas,
                                                  rescale=rescale,
                                                  batch_gt_instances=batch_gt_instances)

        return predictions

    def gen_noise_by_hungarian(self,
                               cls_scores: List[Tensor],
                               bbox_preds: List[Tensor],
                               objectnesses: Optional[List[Tensor]] = None,
                               batch_img_metas: Optional[List[dict]] = None,
                               cfg: Optional[ConfigDict] = None,
                               rescale: bool = True,
                               with_nms: bool = True,
                               batch_gt_instances=None):

        gt_labels_list = []
        for gt_instance in batch_gt_instances:
            gt_labels_list.append(gt_instance.labels.detach())

        # for gen num groups
        num_target_list = [len(bboxes) for bboxes in gt_labels_list]
        max_num_target = max(num_target_list)

        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label
        # cfg.max_per_img = 300

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(featmap_sizes,
                                                                dtype=cls_scores[0].dtype,
                                                                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel() * self.num_base_priors, ), stride)
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        label_embed_list = []
        pos_num_list = []
        neg_num_list = []
        coord_list = []
        results_list = []
        for (bboxes, scores, objectness, img_meta, num_targets,
             gt_instances) in zip(flatten_decoded_bboxes, flatten_cls_scores, flatten_objectness, batch_img_metas,
                                  num_target_list, batch_gt_instances):
            # detach
            bboxes = bboxes.detach()
            scores = scores.detach()
            objectness = objectness.detach()

            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get('yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0 or num_targets == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                label_embed_list.append(torch.zeros((max_num_target * 2, 256), device=bboxes.device))
                coord_list.append(torch.zeros((max_num_target * 2, 4), device=bboxes.device))
                continue

            nms_pre = cfg.get('nms_pre', 900)

            # backup ori scores for hus
            ori_scores = scores.clone()

            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(scores,
                                                                       score_thr,
                                                                       nms_pre,
                                                                       results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(scores, score_thr, nms_pre)

            pred_instances_80 = InstanceData(scores=ori_scores[keep_idxs], labels=labels, bboxes=bboxes[keep_idxs])
            results_list.append(pred_instances_80)

            pred_instances_single = InstanceData(scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    pred_instances_single.bboxes -= pred_instances_single.bboxes.new_tensor(
                        [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
                pred_instances_single.bboxes /= pred_instances_single.bboxes.new_tensor(scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(pred_instances_single)

            pred_instances_single, nms_keep_idx = self._bbox_post_process(results=pred_instances_single,
                                                                          cfg=cfg,
                                                                          rescale=False,
                                                                          with_nms=with_nms,
                                                                          img_meta=img_meta,
                                                                          return_idx=True)

            if nms_keep_idx is not None:
                pred_instances_80 = pred_instances_80[nms_keep_idx][:cfg.max_per_img]

            pred_instances_80.scores = inverse_sigmoid(pred_instances_80.scores, eps=1e-3)
            assign_result = self.dn_assinger.assign(pred_instances=pred_instances_80,
                                                    gt_instances=gt_instances,
                                                    img_meta=img_meta)

            # gen noise
            first_pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()

            fisrt_match_as_pos = pred_instances_single[first_pos_inds]

            all_idxs = np.arange(len(pred_instances_80))
            remain_idxs = torch.tensor(np.setdiff1d(all_idxs, first_pos_inds.cpu().numpy()), device=scores.device)
            remain_pred_instances_80 = pred_instances_80[remain_idxs]
            remain_pred_instances_single = pred_instances_single[remain_idxs]

            second_assign_result = self.dn_assinger.assign(pred_instances=remain_pred_instances_80,
                                                           gt_instances=gt_instances,
                                                           img_meta=img_meta)

            second_pos_inds = torch.nonzero(second_assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
            second_match_as_neg = remain_pred_instances_single[second_pos_inds]

            # diff = max_num_target - num_targets
            diff = max_num_target - fisrt_match_as_pos.bboxes.size(0)
            pos_corrd = torch.cat((fisrt_match_as_pos.bboxes, torch.zeros((diff, 4), device=bboxes.device)))
            neg_corrd = torch.cat((second_match_as_neg.bboxes, torch.zeros((diff, 4), device=bboxes.device)))
            coord = torch.cat((pos_corrd, neg_corrd))

            pos_labbel_embed = torch.cat(
                (self.label_embedding(fisrt_match_as_pos.labels), torch.zeros((diff, 256), device=bboxes.device)))
            neg_labbel_embed = torch.cat(
                (self.label_embedding(second_match_as_neg.labels), torch.zeros((diff, 256), device=bboxes.device)))

            label_embed = torch.cat((pos_labbel_embed, neg_labbel_embed))

            coord[:, 0::2] = coord[:, 0::2] / ori_shape[1]
            coord[:, 1::2] = coord[:, 1::2] / ori_shape[0]

            coord = bbox_xyxy_to_cxcywh(coord)
            coord.clamp_(0.0, 1.0)

            if neg_corrd.size(0) < pos_corrd.size(0):
                _diff = pos_corrd.size(0) - neg_corrd.size(0)
                label_embed = torch.cat((label_embed, torch.zeros((_diff, 256), device=bboxes.device)))
                coord = torch.cat((coord, torch.zeros((_diff, 4), device=bboxes.device)))
                print_log(f'pos neg not fit, diff= {_diff}', self.logger_name, logging.WARNING)

            label_embed_list.append(label_embed)
            coord_list.append(coord)
            pos_num_list.append(first_pos_inds.size(0))
            neg_num_list.append(second_pos_inds.size(0))
        # try:
        return torch.stack(coord_list, 0), torch.stack(label_embed_list, 0), 1
        # except:
        #     print(max_num_target, num_target_list)
        #     print(pos_num_list, neg_num_list)
        #     for _ in coord_list:
        #         print(_.shape[0])

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None,
                           return_idx=False) -> InstanceData:
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        keep_idxs = None

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
        if return_idx:
            return results, keep_idxs

        return results


@MODELS.register_module()
class YOLOv8Head(YOLOv5Head):
    """YOLOv8Head head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(type='mmdet.CrossEntropyLoss',
                                             use_sigmoid=True,
                                             reduction='none',
                                             loss_weight=0.5),
                 loss_bbox: ConfigType = dict(type='IoULoss',
                                              iou_mode='ciou',
                                              bbox_format='xyxy',
                                              reduction='sum',
                                              loss_weight=7.5,
                                              return_iou=False),
                 loss_dfl=dict(type='mmdet.DistributionFocalLoss', reduction='mean', loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        self.assigner_cfg = dict(type='BatchTaskAlignedAssigner',
                                 num_classes=80,
                                 use_ciou=True,
                                 topk=10,
                                 alpha=0.5,
                                 beta=6.0,
                                 eps=1e-09)
        super().__init__(head_module=head_module,
                         prior_generator=prior_generator,
                         bbox_coder=bbox_coder,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        # YOLOv8 doesn't need loss_obj
        self.loss_obj = None
        self.dn_assinger = HungarianAssigner(match_costs=[
            dict(type='FocalLossCost', weight=2.0),
            dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            dict(type='IoUCost', iou_mode='giou', weight=2.0)
        ])

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        self.assigner = TASK_UTILS.build(self.assigner_cfg)

        # Add common attributes to reduce calculation
        self.featmap_sizes_train = None
        self.num_level_priors = None
        self.flatten_priors_train = None
        self.stride_tensor = None

    def loss_by_feat(self,
                     cls_scores: Sequence[Tensor],
                     bbox_preds: Sequence[Tensor],
                     bbox_dist_preds: Sequence[Tensor],
                     batch_gt_instances: Sequence[InstanceData],
                     batch_img_metas: Sequence[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        assert hasattr(self, 'label_embedding')
        
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(self.featmap_sizes_train,
                                                                       dtype=cls_scores[0].dtype,
                                                                       device=cls_scores[0].device,
                                                                       with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4) for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(self.flatten_priors_train[..., :2], flatten_pred_bboxes,
                                                     self.stride_tensor[..., 0])

        assigned_result = self.assigner((flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
                                        flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train, gt_labels,
                                        gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']  # (bs, flatten_preds)

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        # before rescale
        self.noise_result = self.post_yolov8_assign_result(fg_mask_pre_prior, flatten_cls_preds, flatten_pred_bboxes,
                                                           batch_gt_instances, batch_img_metas)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])  # (bs, flatten_preds, 4)
            pred_bboxes_pos = torch.masked_select(flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(pred_bboxes_pos, assigned_bboxes_pos, weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(self.flatten_priors_train[..., :2] / self.stride_tensor,
                                                   assigned_bboxes,
                                                   max_dis=self.head_module.reg_max - 1,
                                                   eps=0.01)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(-1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1, 4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()
        return dict(rpn_loss_cls=loss_cls * num_imgs * world_size,
                    rpn_loss_bbox=loss_bbox * num_imgs * world_size,
                    rpn_loss_dfl=loss_dfl * num_imgs * world_size)

    def gen_noise(self, *args, **kwargs):
        return self.noise_result

    def post_yolov8_assign_result(self, fg_mask_pre_prior, flatten_cls_preds, flatten_pred_bboxes, batch_gt_instances,
                                  batch_img_metas):
        gt_labels_list = []
        for gt_instance in batch_gt_instances:
            gt_labels_list.append(gt_instance.labels.detach())

        # for gen num groups
        num_target_list = [len(bboxes) for bboxes in gt_labels_list]
        max_num_target = max(num_target_list)
        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            label_embed_list = []
            coord_list = []
            for num_targets, prior_mask, single_img_p_bbox, single_img_p_cls, gt_instances, img_meta in zip(
                    num_target_list, fg_mask_pre_prior, flatten_pred_bboxes, flatten_cls_preds, batch_gt_instances,
                    batch_img_metas):

                # ori_shape = img_meta['ori_shape']
                img_shape = img_meta['img_shape']
                # factor = flatten_pred_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(flatten_pred_bboxes.size(0), 1)
                prior_bbox_mask = prior_mask.unsqueeze(-1).repeat([1, 4])  # (bs, flatten_preds, 4)
                prior_scores_mask = prior_mask.unsqueeze(-1).repeat([1, self.num_classes])
                pred_bboxes_pos = torch.masked_select(single_img_p_bbox, prior_bbox_mask).reshape([-1,
                                                                                                   4]).detach()  # xyxy
                pred_scores_pos = torch.masked_select(single_img_p_cls,
                                                      prior_scores_mask).reshape([-1, self.num_classes]).detach()
                pred_lable_pos = pred_scores_pos.argmax(dim=-1).detach()
                if pred_bboxes_pos.shape[0] == 0 or num_targets == 0:
                    empty_results = InstanceData()
                    empty_results.bboxes = pred_bboxes_pos
                    empty_results.scores = pred_scores_pos[:, 0]
                    empty_results.labels = pred_scores_pos[:, 0].int()
                    label_embed_list.append(torch.zeros((max_num_target * 2, 256), device=pred_bboxes_pos.device))
                    coord_list.append(torch.zeros((max_num_target * 2, 4), device=pred_bboxes_pos.device))
                    continue

                pred_instances = InstanceData(scores=pred_scores_pos, labels=pred_lable_pos, bboxes=pred_bboxes_pos)
                assign_result = self.dn_assinger.assign(pred_instances=pred_instances,
                                                        gt_instances=gt_instances,
                                                        img_meta=img_meta)

                # gen noise
                first_pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()

                fisrt_match_as_pos = pred_instances[first_pos_inds]

                all_idxs = np.arange(len(pred_instances))
                remain_idxs = torch.tensor(np.setdiff1d(all_idxs,
                                                        first_pos_inds.cpu().numpy()),
                                           device=pred_scores_pos.device)
                remain_pred_instances = pred_instances[remain_idxs]

                second_assign_result = self.dn_assinger.assign(pred_instances=remain_pred_instances,
                                                               gt_instances=gt_instances,
                                                               img_meta=img_meta)

                second_pos_inds = torch.nonzero(second_assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
                second_match_as_neg = remain_pred_instances[second_pos_inds]

                # diff = max_num_target - num_targets
                diff = max_num_target - fisrt_match_as_pos.bboxes.size(0)
                pos_corrd = torch.cat((fisrt_match_as_pos.bboxes, torch.zeros((diff, 4),
                                                                              device=pred_bboxes_pos.device)))
                neg_corrd = torch.cat((second_match_as_neg.bboxes, torch.zeros((diff, 4),
                                                                               device=pred_bboxes_pos.device)))
                coord = torch.cat((pos_corrd, neg_corrd))

                pos_labbel_embed = torch.cat((self.label_embedding(fisrt_match_as_pos.labels),
                                              torch.zeros((diff, 256), device=pred_bboxes_pos.device)))
                neg_labbel_embed = torch.cat((self.label_embedding(second_match_as_neg.labels),
                                              torch.zeros((diff, 256), device=pred_bboxes_pos.device)))

                label_embed = torch.cat((pos_labbel_embed, neg_labbel_embed))

                coord[:, 0::2] = coord[:, 0::2] / img_shape[1]
                coord[:, 1::2] = coord[:, 1::2] / img_shape[0]
                # coord /= factor

                coord = bbox_xyxy_to_cxcywh(coord)
                coord.clamp_(0.0, 1.0)

                if neg_corrd.size(0) < pos_corrd.size(0):
                    _diff = pos_corrd.size(0) - neg_corrd.size(0)
                    label_embed = torch.cat((label_embed, torch.zeros((_diff, 256), device=pred_bboxes_pos.device)))
                    coord = torch.cat((coord, torch.zeros((_diff, 4), device=pred_bboxes_pos.device)))
                    # print_log(f'pos neg not fit, diff= {_diff}', self.logger_name, logging.WARNING)

                label_embed_list.append(label_embed)
                coord_list.append(coord)
        return torch.stack(coord_list, 0), torch.stack(label_embed_list, 0), 1
