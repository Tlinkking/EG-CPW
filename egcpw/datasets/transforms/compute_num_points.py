# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict

import torch
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ComputeNumPointsInBoxes(BaseTransform):
    """Count LiDAR points inside each GT 3D box.

    This transform is intended for CPU data pipelines. It should be placed
    after point and box augmentations/filtering, and before
    ``Pack3DDetInputs``.

    Box format:
        [x, y, z, w, l, h, yaw]

    For LiDARInstance3DBoxes in MMDetection3D, z is the bottom center of the
    box. The point coordinates and box coordinates must be in the same frame.
    """

    def transform(self, results: Dict) -> Dict:
        points = results['points']
        gt_bboxes_3d = results['gt_bboxes_3d']

        num_gts = len(gt_bboxes_3d)

        if hasattr(points, 'tensor'):
            points_tensor = points.tensor
        else:
            points_tensor = points

        if not isinstance(points_tensor, torch.Tensor):
            points_tensor = torch.as_tensor(points_tensor)

        if hasattr(gt_bboxes_3d, 'tensor'):
            boxes_tensor = gt_bboxes_3d.tensor
        else:
            boxes_tensor = gt_bboxes_3d

        if not isinstance(boxes_tensor, torch.Tensor):
            boxes_tensor = torch.as_tensor(boxes_tensor)

        if points_tensor.ndim != 2 or points_tensor.shape[1] < 3:
            raise ValueError(
                'points must have shape [N, C] with C >= 3, '
                f'but got {tuple(points_tensor.shape)}')

        if boxes_tensor.ndim != 2 or boxes_tensor.shape[1] < 7:
            raise ValueError(
                'gt_bboxes_3d must have shape [M, C] with C >= 7, '
                f'but got {tuple(boxes_tensor.shape)}')

        if num_gts == 0:
            results['num_lidar_pts'] = torch.zeros(
                0, dtype=torch.long, device=boxes_tensor.device)
            return results

        if points_tensor.shape[0] == 0:
            results['num_lidar_pts'] = torch.zeros(
                num_gts, dtype=torch.long, device=boxes_tensor.device)
            return results

        # Use the same device for points and boxes.
        device = boxes_tensor.device
        points_xyz = points_tensor[:, :3].to(
            device=device, dtype=torch.float32)
        boxes_7d = boxes_tensor[:, :7].to(
            device=device, dtype=torch.float32)

        point_counts = []

        for box in boxes_7d:
            cx, cy, cz, width, length, height, yaw = box.unbind()

            if width <= 0 or length <= 0 or height <= 0:
                point_counts.append(
                    torch.zeros((), dtype=torch.long, device=device))
                continue

            dx = points_xyz[:, 0] - cx
            dy = points_xyz[:, 1] - cy
            dz = points_xyz[:, 2] - cz

            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)

            # Transform points into the box local coordinate system.
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy

            inside = (
                (local_x >= -width / 2)
                & (local_x <= width / 2)
                & (local_y >= -length / 2)
                & (local_y <= length / 2)
                & (dz >= 0)
                & (dz <= height)
            )

            point_counts.append(inside.sum().to(torch.long))

        results['num_lidar_pts'] = torch.stack(point_counts)
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__
