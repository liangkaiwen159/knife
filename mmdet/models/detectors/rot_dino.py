# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Dict, List, Tuple, Union, Optional
from torch import Tensor, nn
from mmdet.utils import OptConfigType
from mmdet.registry import MODELS
import torch.nn.functional as F
from mmdet.models.utils import unpack_gt_instances, multi_apply
from .dino import DINO
from ..layers import (CdnQueryGenrator_rpn_rot, DeformableDetrTransformerEncoder, DinoTransformerDecoder,
                      SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from torch.nn.init import normal_
from mmdet.structures import OptSampleList, SampleList
from mmengine.structures import InstanceData
from mmdet.structures.bbox import RotatedBoxes
from mmdet.models.layers.transformer.utils import inverse_sigmoid, act_rot_sigmoid, inverse_act_rot_sigmoid, rot_coord_sigmoid, inverse_rot_coord_sigmoid


@MODELS.register_module()
class Rot_DINO(DINO):

    def __init__(self,
                 *args,
                 use_rpn=False,
                 encoder_reasign=False,
                 num_classes=15,
                 angle_version: str = 'le90',
                 latter: bool = False,
                 **kwargs):
        self.rpn_head_cfg = kwargs.pop('rpn_head', None)
        self.use_rpn = use_rpn
        self.num_classes = num_classes
        self.latter = latter
        self.angle_version = angle_version
        super().__init__(*args, encoder_reasign=encoder_reasign, **kwargs)
        if hasattr(self, 'dn_query_generator'):
            assert 'dn_cfg' in kwargs
            self.dn_query_generator = CdnQueryGenrator_rpn_rot(**kwargs['dn_cfg'])
            self.bbox_coder = self.dn_query_generator.bbox_coder
        else:
            self.bbox_coder = None

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        if self.encoder:
            self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)

        # add bbox_decoder
        setattr(self.decoder, 'angle_version', self.angle_version)
        self.decoder._build_bbox_decoder()

        self.embed_dims = self.decoder.embed_dims
        if self.rpn_head_cfg:
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

        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
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

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
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
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        losses = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples)
        if hasattr(self, 'rpn_loss') and self.use_rpn:
            losses.update(self.rpn_loss)

        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:

        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        results_list = self.bbox_head.predict(**head_inputs_dict,
                                              rescale=rescale,
                                              batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def forward_encoder(self, feat, feat_ori, batch_data_samples, feat_mask: Tensor, feat_pos: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor, valid_ratios: Tensor) -> Dict:

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
                self.rpn_loss = self.rpn_head.loss(feat_ori, batch_data_samples) if self.use_rpn else None
                rpn_results = self.rpn_head.gen_noise(feat_ori, batch_data_samples) if self.use_rpn else None

            else:
                bs = memory.shape[0]
                embed_dim = memory.shape[-1]

                nums_of_levels = len(feat_ori)

                hw = []
                for lvl_idx in range(nums_of_levels):
                    hw.append(feat_ori[lvl_idx].shape[2:])

                resume = memory.permute(0, 2, 1).split([h_i * w_i for h_i, w_i in hw], dim=-1)

                resume = [r.reshape(bs, embed_dim, h_i, w_i) for r, (h_i, w_i) in zip(resume, hw)]
                self.rpn_loss = self.rpn_head.loss(resume, batch_data_samples) if self.use_rpn else None
                rpn_results = self.rpn_head.gen_noise(resume, batch_data_samples) if self.use_rpn else None

        else:
            rpn_results = None
            # self.bbox_head.rpn_results = self.rpn_head.predict(feat_ori, batch_data_samples, rescale=True)

        encoder_outputs_dict = dict(memory=memory,
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
    ) -> Tuple[Dict]:  # sourcery skip: hoist-similar-statement-from-if, hoist-statement-from-if

        bs, _, c = memory.shape
        device = memory.device

        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, memory_mask, spatial_shapes)
        # if not self.use_rpn:
        cls_out_features = self.bbox_head.cls_branches[self.decoder.num_layers].out_features  # int 80
        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory)  # torch.Size([6, 13608, 80])
        enc_outputs_coord_unact = self.bbox_head.reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals[..., :4]
        enc_outputs_rot_unact = self.bbox_head.rot_branches[self.decoder.num_layers](output_memory) + output_proposals[
            ..., 4:]
        if not self.encoder_reasign or not self.training:
            topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        else:
            topk_indices = self.asign_encoder_output(enc_outputs_class,
                                                     torch.cat((enc_outputs_coord_unact, enc_outputs_rot_unact), -1),
                                                     batch_data_samples).permute(1, 0).to(enc_outputs_class.device)
        topk_score = torch.gather(enc_outputs_class, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_rot_unact = torch.gather(enc_outputs_rot_unact, 1, topk_indices.unsqueeze(-1))
        # topk_output_proposals = torch.gather(output_proposals, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))

        topk_coords = torch.cat((topk_coords_unact.sigmoid(), act_rot_sigmoid(topk_rot_unact, self.angle_version)), -1)
        # topk_coords = RotatedBoxes(torch.cat((topk_coords, topk_rot),
        #                                      dim=-1)).regularize_boxes(self.angle_version).to(device)
        # topk_coords = self.bbox_coder.decode(
        #     topk_coords,
        #     torch.cat(
        #         (topk_output_proposals, topk_output_proposals.new_zeros((tuple(topk_output_proposals.shape[:-1]) + (1, )))),
        #         dim=-1))

        topk_coords_unact = inverse_rot_coord_sigmoid(topk_coords, self.angle_version).detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training and hasattr(self, "dn_query_generator"):
            if self.use_rpn:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                    self.dn_query_generator(batch_data_samples, rpn_results)
            else:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                    self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = rot_coord_sigmoid(reference_points, self.angle_version)

        decoder_inputs_dict = dict(query=query, memory=memory, reference_points=reference_points, dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(enc_outputs_class=topk_score,
                                enc_outputs_coord=topk_coords[..., :4],
                                enc_outputs_rot=topk_coords[..., 4:],
                                dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def gen_encoder_output_proposals(self, memory: Tensor, memory_mask: Tensor,
                                     spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:

        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
                -1, keepdim=True) == output_proposals.shape[-1]
        # Do not inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals_rot = output_proposals.new_zeros((tuple(output_proposals.shape[:-1]) + (1, )))
        output_proposals_rot = inverse_act_rot_sigmoid(output_proposals_rot, self.angle_version)
        output_proposals = torch.cat((output_proposals, output_proposals_rot), dim=-1)
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None) -> Dict:

        inter_states, references = self.decoder(query=query,
                                                value=memory,
                                                key_padding_mask=memory_mask,
                                                self_attn_mask=dn_mask,
                                                reference_points=reference_points,
                                                spatial_shapes=spatial_shapes,
                                                level_start_index=level_start_index,
                                                valid_ratios=valid_ratios,
                                                reg_branches=self.bbox_head.reg_branches,
                                                rot_branches=self.bbox_head.rot_branches)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    def forward_transformer(
        self,
        img_feats,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        if isinstance(img_feats, (list, tuple)):
            img_feats_ori, img_feats_channel_map = img_feats
            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats_channel_map, batch_data_samples)
        else:
            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(feat_ori=img_feats_ori,
                                                    **encoder_inputs_dict,
                                                    batch_data_samples=batch_data_samples)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def asign_encoder_output(self, enc_outputs_class: Tensor, enc_outputs_coord_unact: Tensor, batch_data_samples):
        if isinstance(batch_data_samples, list):
            batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
        else:
            batch_gt_instances = batch_data_samples['bboxes_labels']
            batch_img_metas = batch_data_samples['img_metas']
            batch_gt_instances_ignore = None

        chose_topk_indices = multi_apply(self.asign_encoder_output_single, batch_gt_instances, enc_outputs_class,
                                         enc_outputs_coord_unact, batch_img_metas)
        return torch.tensor(chose_topk_indices)

    def asign_encoder_output_single(self, gt_instances, enc_output_class: Tensor, enc_output_coord_unact: Tensor,
                                    img_meta):

        enc_output_coord_act = rot_coord_sigmoid(enc_output_coord_unact, self.angle_version)
        # gt_instances.bboxes.tensor = gt_instances.bboxes.tensor.to('cpu')

        img_h, img_w = img_meta['img_shape']

        bbox_pred = RotatedBoxes(
            self.bbox_head.resize_rot_bbox_tensor(enc_output_coord_act, (float(img_w), float(img_h))))

        pred_instances = InstanceData(scores=enc_output_class, bboxes=bbox_pred)
        num_gts, num_preds = len(gt_instances), len(pred_instances)

        if num_gts == 0 or num_preds == 0:
            # print(f'num_gts: {num_gts}, num_preds: {num_preds}')
            return torch.argsort(enc_output_class.max(-1)[0])[:self.num_queries]

        # 2. compute weighted cost
        cost_list = []
        for match_cost in self.encoder_reasign_match_costs:
            cost = match_cost(pred_instances=pred_instances, gt_instances=gt_instances, img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)
        cost = cost.sum(dim=1)
        return torch.argsort(cost)[:self.num_queries]