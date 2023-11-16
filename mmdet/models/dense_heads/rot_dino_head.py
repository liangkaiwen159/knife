# Copyright (c) OpenMMLab. All rights reserved.
from .dino_head import DINOHead
from mmdet.registry import MODELS
from mmcv.cnn import Linear
import torch.nn as nn
import torch
import copy
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, get_box_tensor
from typing import Dict, List, Tuple
from mmcv.ops import batched_nms
from ..utils import multi_apply
import torch.nn.functional as F
from mmdet.structures.bbox import RotatedBoxes
from mmengine.model import bias_init_with_prob, constant_init
from mmdet.models.layers.transformer.utils import inverse_act_rot_sigmoid, inverse_sigmoid, inverse_rot_coord_sigmoid, rot_coord_sigmoid, act_rot_sigmoid


@MODELS.register_module()
class Rot_DINOHead(DINOHead):

    def __init__(self, *args, angle_version: str = 'le90', anchor_generator=None, **kwargs) -> None:
        super().__init__(*args, anchor_generator=anchor_generator, **kwargs)
        self.angle_version = angle_version

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        rot_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            rot_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
            rot_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        rot_branch.append(Linear(self.embed_dims, 1))
        reg_branch = nn.Sequential(*reg_branch)
        rot_branch = nn.Sequential(*rot_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_pred_layer)])
            self.rot_branches = nn.ModuleList([rot_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList([copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)])
            self.rot_branches = nn.ModuleList([copy.deepcopy(rot_branch) for _ in range(self.num_pred_layer)])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
            for m in self.rot_branches:
                nn.init.constant_(m[-1].bias.data, 0.0)

    def forward(self, hidden_states: Tensor, references: List[Tensor]) -> Tuple[Tensor]:
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_rots = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_rot_coord_sigmoid(references[layer_id], self.angle_version)
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            tmp_rot_preds = self.rot_branches[layer_id](hidden_state)

            assert reference.shape[-1] == 5
            tmp_reg_preds += reference[..., :4]
            tmp_rot_preds += reference[..., 4:]

            outputs_coord = tmp_reg_preds.sigmoid()
            outputs_rot = act_rot_sigmoid(tmp_rot_preds, self.angle_version)
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_rots.append(outputs_rot)
        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_rots = torch.stack(all_layers_outputs_rots)

        return all_layers_outputs_classes, all_layers_outputs_coords, all_layers_outputs_rots

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        outs = self(hidden_states, references)

        predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)

        if hasattr(self, 'rpn_results'):
            _predictions = []
            for prediction, rpn_result in zip(predictions, self.rpn_results):
                results = InstanceData.cat([prediction, rpn_result])
                bboxes = get_box_tensor(results.bboxes)
                det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels,
                                                    dict(type='nms', iou_threshold=0.8))
                results = results[keep_idxs]
                # some nms would reweight the score, such as softnms
                results.scores = det_bboxes[:, -1]
                _predictions.append(results[:300])
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        all_layers_outputs_rots: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.
 
        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        bbox_rots = all_layers_outputs_rots[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            bbox_rot = bbox_rots[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred, bbox_rot, img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                bbox_rots: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:

        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_h, img_w = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
            bbox_rots = bbox_rots[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_rots = bbox_rots[bbox_index]
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_pred
        # det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        # det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        # det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        # det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        # if rescale:
        #     assert img_meta.get('scale_factor') is not None
        #     det_bboxes /= det_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))

        score_threshold_index = scores > 0.01
        
        results = InstanceData()
        results.bboxes = torch.cat((det_bboxes, bbox_rots), dim=-1)[score_threshold_index]
        results.bboxes = self.resize_rot_bbox_tensor(results.bboxes, (float(img_w), float(img_h)))

        results.scores = scores[score_threshold_index]
        results.labels = det_labels[score_threshold_index]
        return results

    def loss(self, hidden_states: Tensor, references: List[Tensor], enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor, enc_outputs_rot: Tensor, batch_data_samples: SampleList,
             dn_meta: Dict[str, int]) -> dict:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord, enc_outputs_rot, batch_gt_instances,
                              batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self,
                     all_layers_cls_scores: Tensor,
                     all_layers_bbox_preds: Tensor,
                     all_layers_bbox_rots: Tensor,
                     enc_cls_scores: Tensor,
                     enc_bbox_preds: Tensor,
                     enc_bbox_rots: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     dn_meta: Dict[str, int],
                     batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:

        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,all_layers_matching_bbox_rots,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds, all_layers_denoising_bbox_rots) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, all_layers_bbox_rots, dn_meta)

        # norm gt rbbox
        # batch_gt_instances_copy = copy.deepcopy(batch_gt_instances)
        batch_gt_instances = self.norm_gt_rbbox(batch_gt_instances)
        if batch_gt_instances_ignore is not None:
            batch_gt_instances_ignore_copy = copy.copy(batch_gt_instances_ignore)
            batch_gt_instances_ignore_copy = self.norm_gt_rbbox(batch_gt_instances_ignore_copy)
        else:
            batch_gt_instances_ignore_copy = None

        loss_dict = self.loss_by_feat_matching(all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                                               all_layers_matching_bbox_rots, batch_gt_instances, batch_img_metas,
                                               batch_gt_instances_ignore_copy)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single_matching(
                    enc_cls_scores, enc_bbox_preds,enc_bbox_rots,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(all_layers_denoising_cls_scores,
                                                                        all_layers_denoising_bbox_preds,
                                                                        all_layers_denoising_bbox_rots,
                                                                        batch_gt_instances=batch_gt_instances,
                                                                        batch_img_metas=batch_img_metas,
                                                                        dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def loss_by_feat_matching(self,
                              all_layers_cls_scores: Tensor,
                              all_layers_bbox_preds: Tensor,
                              all_layers_bbox_rots: Tensor,
                              batch_gt_instances: InstanceList,
                              batch_img_metas: List[dict],
                              batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou = multi_apply(self.loss_by_feat_single_matching,
                                                          all_layers_cls_scores,
                                                          all_layers_bbox_preds,
                                                          all_layers_bbox_rots,
                                                          batch_gt_instances=batch_gt_instances,
                                                          batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single_matching(self, cls_scores: Tensor, bbox_preds: Tensor, bbox_rots: Tensor,
                                     batch_gt_instances: InstanceList, batch_img_metas: List[dict]) -> Tuple[Tensor]:

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        bbox_rots_list = [bbox_rots[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, bbox_rots_list, batch_gt_instances,
                                           batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            # factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h, 1]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factor = (float(img_w), float(img_h))
            factors.append(factor)
        # factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = torch.cat((bbox_preds, bbox_rots), dim=-1)
        # bbox_preds = RotatedBoxes(bbox_preds).regularize_boxes(self.angle_version)

        bbox_preds = self.resize_rot_bbox_tensor(bbox_preds, (float(img_w), float(img_h))).reshape(-1, 5)

        # bboxes = bbox_preds
        # bboxes_gt = bbox_targets

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        #TODO regression L1 loss  解决图像不一样尺寸的问题
        bbox_preds = self.resize_rot_bbox_tensor(bbox_preds, (1 / float(img_w), 1 / float(img_h)))
        bbox_targets = self.resize_rot_bbox_tensor(bbox_targets, (1 / float(img_w), 1 / float(img_h)))

        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos,
                                   img_wh=(float(img_w), float(img_h)))
        return loss_cls, loss_bbox, loss_iou

    def get_targets(self, cls_scores_list: List[Tensor], bbox_preds_list: List[Tensor], bbox_rots_list: List[Tensor],
                    batch_gt_instances: InstanceList, batch_img_metas: List[dict]) -> tuple:

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_targets_single, cls_scores_list, bbox_preds_list, bbox_rots_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor, bbox_rot: Tensor, gt_instances: InstanceData,
                            img_meta: dict) -> tuple:

        img_h, img_w = img_meta['img_shape']
        num_bboxes = bbox_pred.size(0)

        # add rot and norm rbbox
        bbox_pred = torch.cat((bbox_pred, bbox_rot), dim=-1)
        bbox_pred = self.resize_rot_bbox_tensor(bbox_pred, (float(img_h), float(img_w)))
        # bbox_pred = RotatedBoxes(bbox_pred.regularize_boxes(self.angle_version))

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)

        # assigner and sampler
        assign_result = self.assigner.assign(pred_instances=pred_instances,
                                             gt_instances=gt_instances,
                                             img_meta=img_meta)

        gt_bboxes = get_box_tensor(gt_instances.bboxes)
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(get_box_tensor(bbox_pred))
        bbox_weights = torch.zeros_like(get_box_tensor(bbox_pred))
        bbox_weights[pos_inds] = 1.0

        pos_gt_bboxes_normalized = pos_gt_bboxes
        pos_gt_bboxes_targets = pos_gt_bboxes_normalized
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor, all_layers_bbox_preds: Tensor, all_layers_bbox_rots: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:

        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = \
                    all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                    all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_rots = \
                    all_layers_bbox_rots[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                    all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                    all_layers_bbox_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_rots = \
                    all_layers_bbox_rots[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_denoising_bbox_rots = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
            all_layers_matching_bbox_rots = all_layers_bbox_rots
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds, all_layers_matching_bbox_rots,
                all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds, all_layers_denoising_bbox_rots)

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor, all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_bbox_rots: Tensor, batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(self._loss_dn_single,
                           all_layers_denoising_cls_scores,
                           all_layers_denoising_bbox_preds,
                           all_layers_denoising_bbox_rots,
                           batch_gt_instances=batch_gt_instances,
                           batch_img_metas=batch_img_metas,
                           dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor, dn_bbox_rots: Tensor,
                        batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances, batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h, 1]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = torch.cat((dn_bbox_preds, dn_bbox_rots), dim=-1)
        # bbox_preds = RotatedBoxes(bbox_preds).regularize_boxes(self.angle_version)
        bbox_preds = self.resize_rot_bbox_tensor(bbox_preds, (float(img_w), float(img_h))).reshape(-1, 5)

        # bboxes = bbox_preds
        # bboxes_gt = bbox_targets

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        #TODO regression L1 loss 解决图像不一样尺寸的问题
        bbox_preds = self.resize_rot_bbox_tensor(bbox_preds, (1 / float(img_w), 1 / float(img_h)))
        bbox_targets = self.resize_rot_bbox_tensor(bbox_targets, (1 / float(img_w), 1 / float(img_h)))

        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos,
                                   img_wh=(float(img_w), float(img_h)))
        return loss_cls, loss_bbox, loss_iou

    def get_dn_targets(self, batch_gt_instances: InstanceList, batch_img_metas: dict, dn_meta: Dict[str, int]) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_dn_targets_single, batch_gt_instances, batch_img_metas, dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_dn_targets_single(self, gt_instances: InstanceData, img_meta: dict, dn_meta: Dict[str, int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = get_box_tensor(gt_instances.bboxes)
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 5, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 5, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        # factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h, 1]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes
        bbox_targets[pos_inds] = gt_bboxes_normalized.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def norm_gt_rbbox(self, batch_gt_instances: InstanceList):
        return [self.norm_gt_rbbox_single(gt_instances) for gt_instances in batch_gt_instances]

    def norm_gt_rbbox_single(self, gt_instances: InstanceData):
        gt_bboxes = gt_instances.bboxes
        gt_bboxes = RotatedBoxes(gt_bboxes.regularize_boxes(self.angle_version))
        gt_instances.bboxes = gt_bboxes
        return gt_instances

    def resize_rot_bbox_tensor(self, boxes, scale_factor: Tuple[float, float]):
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
        assert len(scale_factor) == 2
        scale_x, scale_y = scale_factor
        ctrs, w, h, t = torch.split(boxes, [2, 1, 1, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)

        # Refer to https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/rotated_boxes.py # noqa
        # rescale centers
        ctrs = ctrs * ctrs.new_tensor([scale_x, scale_y])
        # rescale width and height
        w = w * torch.sqrt((scale_x * cos_value)**2 + (scale_y * sin_value)**2)
        h = h * torch.sqrt((scale_x * sin_value)**2 + (scale_y * cos_value)**2)
        # recalculate theta
        t = torch.atan2(scale_x * sin_value, scale_y * cos_value)
        boxes = torch.cat([ctrs, w, h, t], dim=-1)

        return boxes