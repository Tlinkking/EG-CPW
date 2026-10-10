import torch
from mmcv.transforms import BaseTransform

# from mmdet3d.datasets.transforms import Pack3DDetInputs
from .formating import Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS


# @TRANSFORMS.register_module()
# class ComputeNumPointsInBoxes(BaseTransform):
#     """Compute the number of points in every final GT 3D box on CPU.

#     This transform must be placed after all transforms that modify points
#     or boxes, including DB sampling, geometric augmentation, and filtering.
#     """

#     def transform(self, results):
#         if 'points' not in results:
#             raise KeyError('results does not contain "points".')

#         if 'gt_bboxes_3d' not in results:
#             raise KeyError('results does not contain "gt_bboxes_3d".')

#         if 'gt_labels_3d' not in results:
#             raise KeyError('results does not contain "gt_labels_3d".')

#         points = results['points']
#         boxes = results['gt_bboxes_3d']
#         labels = results['gt_labels_3d']

#         points_xyz = points.tensor[:, :3].to(dtype=torch.float32)
#         box_tensor = boxes.tensor[:, :7].to(dtype=torch.float32)

#         if points_xyz.is_cuda or box_tensor.is_cuda:
#             raise RuntimeError(
#                 'ComputeNumPointsInBoxes must run on CPU in the '
#                 'DataLoader pipeline.'
#             )

#         num_points = points_xyz.shape[0]
#         num_boxes = box_tensor.shape[0]

#         if num_boxes != len(labels):
#             raise RuntimeError(
#                 'The number of GT boxes and labels does not match: '
#                 f'{num_boxes} vs {len(labels)}'
#             )

#         if num_boxes == 0:
#             results['num_lidar_pts'] = torch.zeros(
#                 (0,), dtype=torch.long
#             )
#             return results

#         # LiDARInstance3DBoxes uses bottom-centered z internally.
#         # gravity_center gives the actual geometric center and makes the
#         # vertical test consistent with the CUDA point-in-box operator.
#         centers = boxes.gravity_center[:, :3].to(dtype=torch.float32)
#         dimensions = box_tensor[:, 3:6]
#         yaws = box_tensor[:, 6]

#         point_xy = points_xyz[:, :2]
#         point_z = points_xyz[:, 2]

#         # offsets[i, j] is point i relative to box j in global XY space.
#         offsets = point_xy[:, None, :] - centers[None, :, :2]

#         cos_yaw = torch.cos(yaws)[None, :]
#         sin_yaw = torch.sin(yaws)[None, :]

#         # Rotate global offsets by -yaw into every box's local frame.
#         local_x = offsets[..., 0] * cos_yaw + offsets[..., 1] * sin_yaw
#         local_y = -offsets[..., 0] * sin_yaw + offsets[..., 1] * cos_yaw

#         half_length = dimensions[:, 0][None, :] * 0.5
#         half_width = dimensions[:, 1][None, :] * 0.5
#         half_height = dimensions[:, 2][None, :] * 0.5

#         inside_x = local_x.abs() <= half_length
#         inside_y = local_y.abs() <= half_width
#         inside_z = (
#             (point_z[:, None] - centers[None, :, 2]).abs()
#             <= half_height
#         )

#         point_box_mask = inside_x & inside_y & inside_z
#         num_lidar_pts = point_box_mask.sum(dim=0).to(dtype=torch.long)

#         if len(num_lidar_pts) != num_boxes:
#             raise RuntimeError(
#                 'The number of point counts and GT boxes does not match: '
#                 f'{len(num_lidar_pts)} vs {num_boxes}'
#             )

#         if num_lidar_pts.numel() > 0:
#             max_count = int(num_lidar_pts.max().item())

#             if max_count > num_points:
#                 raise RuntimeError(
#                     'A box contains more points than the whole point cloud: '
#                     f'max_box_count={max_count}, '
#                     f'total_points={num_points}'
#                 )

#         results['num_lidar_pts'] = num_lidar_pts
#         return results


@TRANSFORMS.register_module()
class Pack3DDetInputsWithPointCounts(Pack3DDetInputs):
    """Pack the computed point count into gt_instances_3d."""

    def transform(self, results):
        packed_results = super().transform(results)

        if 'num_lidar_pts' not in results:
            raise KeyError(
                'results does not contain "num_lidar_pts". '
                'ComputeNumPointsInBoxes must run before packing.'
            )

        data_sample = packed_results['data_samples']

        if not hasattr(data_sample, 'gt_instances_3d'):
            raise RuntimeError(
                'Packed data sample does not contain gt_instances_3d.'
            )

        num_lidar_pts = results['num_lidar_pts']

        if not isinstance(num_lidar_pts, torch.Tensor):
            num_lidar_pts = torch.as_tensor(
                num_lidar_pts,
                dtype=torch.long,
            )

        num_lidar_pts = num_lidar_pts.to(dtype=torch.long)

        num_boxes = len(data_sample.gt_instances_3d.bboxes_3d)

        if len(num_lidar_pts) != num_boxes:
            raise RuntimeError(
                'The number of point counts and packed GT boxes '
                f'does not match: {len(num_lidar_pts)} vs {num_boxes}'
            )

        data_sample.gt_instances_3d.num_lidar_pts = num_lidar_pts
        return packed_results
