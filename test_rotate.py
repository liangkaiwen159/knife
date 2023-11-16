from mmcv.ops import diff_iou_rotated_2d

import torch

if __name__ == '__main__':
    pred = torch.tensor([[40.0, 50, 20, 20, 0.8], \
                         [40.0, 50, 20, 20, 1], \
                         [40.0, 50, 20, 20, 0.7]]).to('cuda:0')
    gt = torch.tensor([[40.0, 50, 20, 20, 1], \
                       [40.0, 50, 20, 20, 0.8]]).to('cuda:0')
    num_pred = pred.size(0)
    num_gt = gt.size(0)
    pred = pred[:, None].repeat(1, num_gt, 1).reshape(-1, 5)
    gt = gt[None].repeat(num_pred, 1, 1).reshape(-1, 5)
    print(diff_iou_rotated_2d(pred[None], gt[None]).reshape(num_pred, num_gt).shape)
    print(diff_iou_rotated_2d(pred[None], gt[None]).squeeze(0).reshape(num_pred, num_gt).shape)