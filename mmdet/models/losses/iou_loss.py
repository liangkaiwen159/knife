# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps
from .utils import weighted_loss
import math
from typing import Optional, Tuple, Union

from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.structures.bbox import HorizontalBoxes
from mmcv.ops import diff_iou_rotated_2d, box_iou_rotated

try:
    from mmcv.ops import diff_iou_rotated_2d
except:  # noqa: E722
    diff_iou_rotated_2d = None


def bbox_overlaps_yolo(pred: torch.Tensor,
                       target: torch.Tensor,
                       iou_mode: str = 'ciou',
                       bbox_format: str = 'xywh',
                       siou_theta: float = 4.0,
                       eps: float = 1e-7) -> torch.Tensor:
    r"""Calculate overlap between two set of bboxes.
    `Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
    difference in the way the alpha parameter is computed.

    mmdet version:
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    YOLOv5 version:
        alpha = v / (v - ious + (1 + eps)

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        siou_theta (float): siou_theta for SIoU when calculate shape cost.
            Defaults to 4.0.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: shape (n, ).
    """
    assert iou_mode in ('iou', 'ciou', 'giou', 'siou')
    assert bbox_format in ('xyxy', 'xywh')
    if bbox_format == 'xywh':
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)

    bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
    bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
    bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
    bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]

    # Overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) -
               torch.max(bbox1_x1, bbox2_x1)).clamp(0) * \
              (torch.min(bbox1_y2, bbox2_y2) -
               torch.max(bbox1_y1, bbox2_y1)).clamp(0)

    # Union
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = (w1 * h1) + (w2 * h2) - overlap + eps

    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
    enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    enclose_w = enclose_wh[..., 0]  # cw
    enclose_h = enclose_wh[..., 1]  # ch

    if iou_mode == 'ciou':
        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                           (bbox1_y1 + bbox1_y2))**2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)

        # Width and height ratio (v)
        wh_ratio = (4 / (math.pi**2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

        # CIoU
        ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

    elif iou_mode == 'giou':
        # GIoU = IoU - ( (A_c - union) / A_c )
        convex_area = enclose_w * enclose_h + eps  # convex area (A_c)
        ious = ious - (convex_area - union) / convex_area

    elif iou_mode == 'siou':
        # SIoU: https://arxiv.org/pdf/2205.12740.pdf
        # SIoU = IoU - ( (Distance Cost + Shape Cost) / 2 )

        # calculate sigma (σ):
        # euclidean distance between bbox2(pred) and bbox1(gt) center point,
        # sigma_cw = b_cx_gt - b_cx
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        # sigma_ch = b_cy_gt - b_cy
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        # sigma = √( (sigma_cw ** 2) - (sigma_ch ** 2) )
        sigma = torch.pow(sigma_cw**2 + sigma_ch**2, 0.5)

        # choose minimize alpha, sin(alpha)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha,
                                sin_beta)

        # Angle cost = 1 - 2 * ( sin^2 ( arcsin(x) - (pi / 4) ) )
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # Distance cost = Σ_(t=x,y) (1 - e ^ (- γ ρ_t))
        rho_x = (sigma_cw / enclose_w)**2  # ρ_x
        rho_y = (sigma_ch / enclose_h)**2  # ρ_y
        gamma = 2 - angle_cost  # γ
        distance_cost = (1 - torch.exp(-1 * gamma * rho_x)) + (
            1 - torch.exp(-1 * gamma * rho_y))

        # Shape cost = Ω = Σ_(t=w,h) ( ( 1 - ( e ^ (-ω_t) ) ) ^ θ )
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # ω_w
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # ω_h
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w),
                               siou_theta) + torch.pow(
                                   1 - torch.exp(-1 * omiga_h), siou_theta)

        ious = ious - ((distance_cost + shape_cost) * 0.5)

    return ious.clamp(min=-1.0, max=1.0)


@weighted_loss
def iou_loss(pred: Tensor,
             target: Tensor,
             linear: bool = False,
             mode: str = 'log',
             eps: float = 1e-6) -> Tensor:
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in '
                      'iou_loss is deprecated, please use "mode=`linear`" '
                      'instead.')
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)

    if fp16:
        ious = ious.to(torch.float16)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


@weighted_loss
def rotated_iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
    rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')

    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')

    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    # ious = box_iou_rotated(pred, target)
    ious = ious.squeeze(0).clamp(min=eps)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


@weighted_loss
def bounded_iou_loss(pred: Tensor,
                     target: Tensor,
                     beta: float = 0.2,
                     eps: float = 1e-3) -> Tensor:
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        beta (float, optional): Beta parameter in smoothl1.
        eps (float, optional): Epsilon to avoid NaN values.

    Return:
        Tensor: Loss tensor.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    # view(..., -1) does not work for empty tensor
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).flatten(1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss


@weighted_loss
def giou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)

    if fp16:
        gious = gious.to(torch.float16)

    loss = 1 - gious
    return loss


@weighted_loss
def diou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


@weighted_loss
def ciou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v)

    # CIoU
    cious = ious - (rho2 / c2 + alpha * v)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss


@weighted_loss
def eiou_loss(pred: Tensor,
              target: Tensor,
              smooth_point: float = 0.1,
              eps: float = 1e-7) -> Tensor:
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        smooth_point (float): hyperparameter, default is 0.1.
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # extent top left
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)

    # intersection coordinates
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    # extra
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)

    # Intersection
    intersection = (ix2 - ex1) * (iy2 - ey1) + (xmin - ex1) * (ymin - ey1) - (
        ix1 - ex1) * (ymax - ey1) - (xmax - ex1) * (
            iy1 - ey1)
    # Union
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (
        ty2 - ty1) - intersection + eps
    # IoU
    ious = 1 - (intersection / union)

    # Smooth-EIoU
    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * (ious**2) / smooth_point + (1 - smooth_sign) * (
        ious - 0.5 * smooth_point)
    return loss


@weighted_loss
def siou_loss(pred, target, eps=1e-7, neg_gamma=False):
    r"""`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    # modified clamp threshold zero to eps to avoid NaN
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=eps)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # angle cost
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps

    sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)

    # distance cost
    rho_x = (s_cw / cw)**2
    rho_y = (s_ch / ch)**2

    # `neg_gamma=True` follows original implementation in paper
    # but setting `neg_gamma=False` makes training more stable.
    gamma = angle_cost - 2 if neg_gamma else 2 - angle_cost
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    # shape cost
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
        1 - torch.exp(-1 * omiga_h), 4)

    # SIoU
    sious = ious - 0.5 * (distance_cost + shape_cost)
    loss = 1 - sious.clamp(min=-1.0, max=1.0)
    return loss


@MODELS.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear: bool = False,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 mode: str = 'log') -> None:
        super().__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight
                is not None) and (not torch.any(weight > 0)) and (reduction
                                                                  != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class BoundedIoULoss(nn.Module):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        beta (float, optional): Beta parameter in smoothl1.
        eps (float, optional): Epsilon to avoid NaN values.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 beta: float = 0.2,
                 eps: float = 1e-3,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * bounded_iou_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class GIoULoss(nn.Module):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class DIoULoss(nn.Module):
    r"""Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * diou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class CIoULoss(nn.Module):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * ciou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class EIoULoss(nn.Module):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 smooth_point: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * eiou_loss(
            pred,
            target,
            weight,
            smooth_point=self.smooth_point,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class SIoULoss(nn.Module):
    r"""`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 neg_gamma: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.neg_gamma = neg_gamma

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
    
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * siou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            neg_gamma=self.neg_gamma,
            **kwargs)
        return loss

class IoULossYolo(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    Args:
        iou_mode (str): Options are "ciou".
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        return_iou (bool): If True, return loss and iou.
    """

    def __init__(self,
                 iou_mode: str = 'ciou',
                 bbox_format: str = 'xywh',
                 eps: float = 1e-7,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 return_iou: bool = True):
        super().__init__()
        assert bbox_format in ('xywh', 'xyxy')
        assert iou_mode in ('ciou', 'siou', 'giou')
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def bbox_overlaps(self,
                      pred: torch.Tensor,
                      target: torch.Tensor,
                      iou_mode: str = 'ciou',
                      bbox_format: str = 'xywh',
                      siou_theta: float = 4.0,
                      eps: float = 1e-7) -> torch.Tensor:
        r"""Calculate overlap between two set of bboxes.
        `Implementation of paper `Enhancing Geometric Factors into
        Model Learning and Inference for Object Detection and Instance
        Segmentation <https://arxiv.org/abs/2005.03572>`_.

        In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
        difference in the way the alpha parameter is computed.

        mmdet version:
            alpha = (ious > 0.5).float() * v / (1 - ious + v)
        YOLOv5 version:
            alpha = v / (v - ious + (1 + eps)

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
                or (x, y, w, h),shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
                Defaults to "ciou".
            bbox_format (str): Options are "xywh" and "xyxy".
                Defaults to "xywh".
            siou_theta (float): siou_theta for SIoU when calculate shape cost.
                Defaults to 4.0.
            eps (float): Eps to avoid log(0).

        Returns:
            Tensor: shape (n, ).
        """
        assert iou_mode in ('iou', 'ciou', 'giou', 'siou')
        assert bbox_format in ('xyxy', 'xywh')
        if bbox_format == 'xywh':
            pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
            target = HorizontalBoxes.cxcywh_to_xyxy(target)

        bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
        bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
        bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
        bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]

        # Overlap
        overlap = (torch.min(bbox1_x2, bbox2_x2) -
                torch.max(bbox1_x1, bbox2_x1)).clamp(0) * \
                (torch.min(bbox1_y2, bbox2_y2) -
                torch.max(bbox1_y1, bbox2_y1)).clamp(0)

        # Union
        w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
        w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
        union = (w1 * h1) + (w2 * h2) - overlap + eps

        h1 = bbox1_y2 - bbox1_y1 + eps
        h2 = bbox2_y2 - bbox2_y1 + eps

        # IoU
        ious = overlap / union

        # enclose area
        enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
        enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        enclose_w = enclose_wh[..., 0]  # cw
        enclose_h = enclose_wh[..., 1]  # ch

        if iou_mode == 'ciou':
            # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

            # calculate enclose area (c^2)
            enclose_area = enclose_w**2 + enclose_h**2 + eps

            # calculate ρ^2(b_pred,b_gt):
            # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
            # center point, because bbox format is xyxy -> left-top xy and
            # right-bottom xy, so need to / 4 to get center point.
            rho2_left_item = ((bbox2_x1 + bbox2_x2) -
                              (bbox1_x1 + bbox1_x2))**2 / 4
            rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                               (bbox1_y1 + bbox1_y2))**2 / 4
            rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)

            # Width and height ratio (v)
            wh_ratio = (4 / (math.pi**2)) * torch.pow(
                torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

            with torch.no_grad():
                alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

            # CIoU
            ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

        elif iou_mode == 'giou':
            # GIoU = IoU - ( (A_c - union) / A_c )
            convex_area = enclose_w * enclose_h + eps  # convex area (A_c)
            ious = ious - (convex_area - union) / convex_area

        elif iou_mode == 'siou':
            # SIoU: https://arxiv.org/pdf/2205.12740.pdf
            # SIoU = IoU - ( (Distance Cost + Shape Cost) / 2 )

            # calculate sigma (σ):
            # euclidean distance between bbox2(pred) and bbox1(gt) center point,
            # sigma_cw = b_cx_gt - b_cx
            sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 +
                                                    bbox1_x2) / 2 + eps
            # sigma_ch = b_cy_gt - b_cy
            sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 +
                                                    bbox1_y2) / 2 + eps
            # sigma = √( (sigma_cw ** 2) - (sigma_ch ** 2) )
            sigma = torch.pow(sigma_cw**2 + sigma_ch**2, 0.5)

            # choose minimize alpha, sin(alpha)
            sin_alpha = torch.abs(sigma_ch) / sigma
            sin_beta = torch.abs(sigma_cw) / sigma
            sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4),
                                    sin_alpha, sin_beta)

            # Angle cost = 1 - 2 * ( sin^2 ( arcsin(x) - (pi / 4) ) )
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

            # Distance cost = Σ_(t=x,y) (1 - e ^ (- γ ρ_t))
            rho_x = (sigma_cw / enclose_w)**2  # ρ_x
            rho_y = (sigma_ch / enclose_h)**2  # ρ_y
            gamma = 2 - angle_cost  # γ
            distance_cost = (1 - torch.exp(-1 * gamma * rho_x)) + (
                1 - torch.exp(-1 * gamma * rho_y))

            # Shape cost = Ω = Σ_(t=w,h) ( ( 1 - ( e ^ (-ω_t) ) ) ^ θ )
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # ω_w
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # ω_h
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w),
                                   siou_theta) + torch.pow(
                                       1 - torch.exp(-1 * omiga_h), siou_theta)

            ious = ious - ((distance_cost + shape_cost) * 0.5)

        return ious.clamp(min=-1.0, max=1.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[Union[str, bool]] = None
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
                or (x, y, w, h),shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            loss or tuple(loss, iou):
        """

        reduction = (
            reduction_override if reduction_override else self.reduction)

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        iou = self.bbox_overlaps(
            pred,
            target,
            iou_mode=self.iou_mode,
            bbox_format=self.bbox_format,
            eps=self.eps)
        loss = self.loss_weight * weight_reduce_loss(1.0 - iou, weight,
                                                     reduction, avg_factor)

        if self.return_iou:
            return loss, iou
        else:
            return loss


@MODELS.register_module()
class RotatedIoULoss(nn.Module):
    """RotatedIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(RotatedIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight
                is not None) and (not torch.any(weight > 0)) and (reduction
                                                                  != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rotated_iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
