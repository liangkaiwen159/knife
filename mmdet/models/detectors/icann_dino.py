# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Dict, List, Tuple, Union
from torch import Tensor, nn
from mmdet.utils import OptConfigType
from mmdet.registry import MODELS
import torch.nn.functional as F
from mmengine.logging import MMLogger, print_log
from .dino import DINO
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from torch.nn.init import normal_
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.models.utils import unpack_gt_instances, multi_apply
from mmengine.structures import InstanceData
from mmdet.models.task_modules.assigners.assign_result import AssignResult


@MODELS.register_module()
class ICANN_DINO(DINO):

    def __init__(self,
                 *args,
                 use_rpn=False,
                 encoder_reasign=False,
                 num_classes=80,
                 latter: bool = False,
                 **kwargs):
        self.rpn_head_cfg = kwargs.pop('rpn_head', None)
        self.use_rpn = use_rpn
        self.num_classes = num_classes
        self.latter = latter
        super().__init__(*args, encoder_reasign=encoder_reasign, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        if self.encoder:
            self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        if self.rpn_head_cfg and self.use_rpn:
            self.rpn_head = MODELS.build(self.rpn_head_cfg)
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)


    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.encoder:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)
        
        if self.use_rpn:
            setattr(self.rpn_head, 'label_embedding',
                    self.dn_query_generator.label_embedding)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        if hasattr(self, 'rpn_loss') and self.use_rpn:
            losses.update(self.rpn_loss)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def forward_encoder(self, feat, feat_ori, batch_data_samples,
                        feat_mask: Tensor, feat_pos: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        if self.encoder:
            memory = self.encoder(  # torch.Size([2, 12650, 256])
                query=feat,
                query_pos=feat_pos,
                key_padding_mask=feat_mask,  # for self_attn
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios)
        else:
            memory = feat + feat_pos

        if self.training:
            if not self.latter:
                self.rpn_loss = self.rpn_head.loss(
                    feat_ori, batch_data_samples) if self.use_rpn else None
                rpn_results = self.rpn_head.gen_noise(
                    feat_ori, batch_data_samples) if self.use_rpn else None

            else:
                bs = memory.shape[0]
                embed_dim = memory.shape[-1]

                nums_of_levels = len(feat_ori)

                hw = []
                for lvl_idx in range(nums_of_levels):
                    hw.append(feat_ori[lvl_idx].shape[2:])

                resume = memory.permute(0, 2, 1).split(
                    [h_i * w_i for h_i, w_i in hw], dim=-1)

                resume = [
                    r.reshape(bs, embed_dim, h_i, w_i)
                    for r, (h_i, w_i) in zip(resume, hw)
                ]
                self.rpn_loss = self.rpn_head.loss(
                    resume, batch_data_samples) if self.use_rpn else None
                rpn_results = self.rpn_head.gen_noise(
                    resume, batch_data_samples) if self.use_rpn else None

        else:
            rpn_results = None
            # self.bbox_head.rpn_results = self.rpn_head.predict(feat_ori, batch_data_samples, rescale=True)

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            rpn_results=rpn_results)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        rpn_results: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[
            Dict]:  # sourcery skip: hoist-similar-statement-from-if, hoist-statement-from-if

        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        # if not self.use_rpn:
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features  # int 80
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)  # torch.Size([6, 13608, 80])
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        if not self.encoder_reasign:
            topk_indices = torch.topk(
                enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        else:
            topk_indices = self.asign_encoder_output(
                enc_outputs_class, enc_outputs_coord_unact,
                batch_data_samples).permute(1, 0).to(enc_outputs_class.device).contiguous()
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            if self.use_rpn:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                    self.dn_query_generator(batch_data_samples, rpn_results)
            else:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                    self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()
        # else:
        #     enc_outputs_class, enc_outputs_obj, enc_outputs_coord_unact = rpn_results
        #     enc_outputs_coord_unact = enc_outputs_coord_unact + output_proposals
        #     topk_indices = torch.topk(enc_outputs_obj.max(-1)[0], k=self.num_queries, dim=1)[1]
        #     topk_score = torch.gather(enc_outputs_class, 1, topk_indices.unsqueeze(-1).repeat(1, 1, self.num_classes))
        #     topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        #     topk_coords = topk_coords_unact.sigmoid()
        #     topk_coords_unact = topk_coords_unact.detach()
        #     query = self.query_embedding.weight[:, None, :]
        #     query = query.repeat(1, bs, 1).transpose(0, 1)
        #     if self.training:
        #         dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
        #             self.dn_query_generator(batch_data_samples)
        #         query = torch.cat([dn_label_query, query], dim=1)
        #         reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        #     else:
        #         reference_points = topk_coords_unact
        #         dn_mask, dn_meta = None, None
        #     reference_points = reference_points.sigmoid()
        # else:
        #     enc_outputs_coord, enc_outputs_class, nums_of_positive = rpn_results
        #     topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        #     topk_score = torch.gather(enc_outputs_class, 1, topk_indices.unsqueeze(-1).repeat(1, 1, self.num_classes))
        #     enc_outputs_coord = torch.gather(enc_outputs_coord, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        #     topk_coords = enc_outputs_coord.detach()
        #     query = self.query_embedding.weight[:, None, :]
        #     query = query.repeat(1, bs, 1).transpose(0, 1)
        #     if self.training:
        #         dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
        #             self.dn_query_generator(batch_data_samples)
        #         query = torch.cat([dn_label_query, query], dim=1)
        #         reference_points = torch.cat([dn_bbox_query.sigmoid(), enc_outputs_coord], dim=1)
        #     else:
        #         reference_points = -torch.log(1 / (enc_outputs_coord + 1e-8) - 1)
        #         dn_mask, dn_meta = None, None

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_transformer(
        self,
        img_feats,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        if isinstance(img_feats, (list, tuple)):
            img_feats_ori, img_feats_channel_map = img_feats
            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
                img_feats_channel_map, batch_data_samples)
        else:
            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
                img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            feat_ori=img_feats_ori,
            **encoder_inputs_dict,
            batch_data_samples=batch_data_samples)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict
