# egcpw/ — EG-CPW core implementation (overlay)

These are the only MMDetection3D 1.3.0 files that EG-CPW adds or
modifies. Copy them into a stock `mmdetection3d` v1.3.0 source tree at
the same relative paths.

| File (relative to mmdetection3d root) | Type | What it does |
|---|---|---|
| `mmdet3d/models/dense_heads/centerpoint_head.py` | **replaces** stock file | Adds the evidence machinery to `CenterHead`: per-class evidence score `e_i = beta * z_count + (1-beta) * z_density`, the `evidence_radius` config block, `mode='radius'` (dynamic Gaussian radius), `mode='peak_weight'` (center-peak positive-loss weighting with `w_i = clip(1 + lambda * clip(-e_i, 0, q_max), 1, omega_max)`), and evidence statistics logging. Gaussian targets, box losses, decoder, NMS, and inference are unchanged. |
| `mmdet3d/datasets/transforms/compute_num_points.py` | **new** | `ComputeNumPointsInBoxes` transform: counts the enclosed LiDAR points of every final GT 3D box on CPU (after all augmentation), producing `num_lidar_pts`. |
| `mmdet3d/datasets/transforms/evidence_transforms.py` | **new** | `Pack3DDetInputsWithPointCounts`: packs `num_lidar_pts` into the training data samples so the head can read per-instance point counts. |
| `mmdet3d/datasets/transforms/formating.py` | **replaces** stock file | Adds `num_lidar_pts` to the packed keys of `Pack3DDetInputs`. |
| `mmdet3d/datasets/transforms/__init__.py` | **replaces** stock file | Registers the two new transforms. |

No other mmdet3d files are modified. In particular, the focal loss is
the unmodified `GaussianFocalLoss` from mmdet 3.2.0 — the peak weighting
is applied to the positive-branch loss contribution inside
`CenterHead.loss_by_feat`, not by editing the loss class.

Quick sanity check after overlaying:

```bash
python -m py_compile \
  mmdet3d/models/dense_heads/centerpoint_head.py \
  mmdet3d/datasets/transforms/__init__.py \
  mmdet3d/datasets/transforms/compute_num_points.py \
  mmdet3d/datasets/transforms/evidence_transforms.py \
  mmdet3d/datasets/transforms/formating.py
```
