# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch.nn as nn
from mmcv.cnn import Linear

from mmdet.registry import MODELS
from .deformable_detr_head import DeformableDETRHead
from mmdet.utils import InstanceList, OptInstanceList
from torch import Tensor
from typing import Dict, List, Tuple
import torch


@MODELS.register_module()
class ICANN_DeforDETRHead(DeformableDETRHead):

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        fc_cls_for_rpn = Linear(self.cls_out_channels, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList([copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)])

    def loss_by_feat(self,
                     all_layers_cls_scores: Tensor,
                     all_layers_bbox_preds: Tensor,
                     enc_cls_scores: Tensor,
                     enc_bbox_preds: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        loss_dict = super(DeformableDETRHead,
                          self).loss_by_feat(all_layers_cls_scores, all_layers_bbox_preds, batch_gt_instances,
                                             batch_img_metas, batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            proposal_gt_instances = copy.deepcopy(batch_gt_instances)
            # 80个类别有意义 注释掉 不给0
            # for i in range(len(proposal_gt_instances)):
            #     proposal_gt_instances[i].labels = torch.zeros_like(proposal_gt_instances[i].labels)
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=proposal_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        return loss_dict
