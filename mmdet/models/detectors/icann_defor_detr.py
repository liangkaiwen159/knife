# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Dict, Optional, Tuple
from torch import Tensor, nn
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmdet.registry import MODELS
from mmengine.model import xavier_init
import torch.nn.functional as F
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerDecoder, DinoTransformerDecoder,
                      SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from torch.nn.init import normal_
import math, time


@MODELS.register_module()
class ICANN_Defor_Detr(DeformableDETR):

    def __init__(self, *args, use_rpn=False, **kwargs):
        self.rpn_head_cfg = kwargs.pop('rpn_head', None)

        self.use_rpn = use_rpn
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.decoder = DeformableDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        if self.rpn_head_cfg:
            self.rpn_head = MODELS.build(self.rpn_head_cfg)

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            if not self.use_rpn:
                self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
                self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            if not self.use_rpn:
                nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)

    def forward_encoder(self, img_feats, batch_data_samples, feat: Tensor, feat_mask: Tensor, feat_pos: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor, valid_ratios: Tensor) -> Dict:
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
        # remove DINO encoder
        # memory = self.encoder( # torch.Size([2, 12650, 256])
        #     query=feat,
        #     query_pos=feat_pos,
        #     key_padding_mask=feat_mask,  # for self_attn
        #     spatial_shapes=spatial_shapes,
        #     level_start_index=level_start_index,
        #     valid_ratios=valid_ratios)
        if self.use_rpn:
            rpn_results = self.rpn_head(img_feats, batch_data_samples)
        else:
            rpn_results = None
        memory = feat + feat_pos
        encoder_outputs_dict = dict(memory=memory,
                                    memory_mask=feat_mask,
                                    spatial_shapes=spatial_shapes,
                                    rpn_results=rpn_results)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor, spatial_shapes: Tensor,
                    rpn_results: Tensor) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        batch_size, _, c = memory.shape
        if self.as_two_stage and not self.use_rpn:
            # TODO 修改proposals生成方式
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[self.decoder.num_layers](
                output_memory) + output_proposals
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            topk_proposals = torch.topk(enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            pos_trans_out = self.pos_trans_fc(self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            query_pos, query = torch.split(pos_trans_out, c, dim=2)

        elif self.use_rpn:
            enc_outputs_coord, enc_outputs_class, nums_of_positive = rpn_results
            reference_points = enc_outputs_coord
            pos_trans_out = self.pos_trans_fc(self.get_proposal_pos_embed_without_sigmoid(enc_outputs_coord.detach()))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)

            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            enc_outputs_class, enc_outputs_coord = None, None
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(query=query, query_pos=query_pos, memory=memory, reference_points=reference_points)
        head_inputs_dict = dict(enc_outputs_class=enc_outputs_class,
                                enc_outputs_coord=enc_outputs_coord) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_transformer(self, img_feats: Tuple[Tensor], batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(img_feats, batch_data_samples, **encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    @staticmethod
    def get_proposal_pos_embed_without_sigmoid(proposals: Tensor,
                                               num_pos_feats: int = 128,
                                               temperature: int = 10000) -> Tensor:
        """Get the position embedding of the proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos