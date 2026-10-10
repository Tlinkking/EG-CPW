"""Dump predictions + per-GT evidence diagnostics for one checkpoint.

Runs a trained CenterPoint checkpoint on the nuScenes validation split with
the exact val pipeline from the controlled config, and saves a pkl with:

- preds: per sample token, LiDAR-frame boxes (N,9), scores, labels;
- gts: per sample token, per-instance table with class id, num_lidar_pts,
  box size/volume, LiDAR-frame center and lidar_range;
- gt_hm_score: sigmoid probability of the predicted heatmap at the GT
  center cell (same class channel, same feature-map mapping as training
  get_targets_single), used only as an auxiliary center-confidence
  diagnostic.

This script performs inference only; it does not modify any model, loss,
pipeline or config file.
"""

import argparse

import numpy as np
import pickle
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner.checkpoint import load_checkpoint
from torch.utils.data import DataLoader

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules

register_all_modules()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/centerpoint/mycenterpoint_nus_evidence_controlled.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--val-info',
        default='/home/bjtc/data/datasets/nuScenes/nuscenes_infos_val.pkl')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    class_names = list(cfg.val_dataloader.dataset.metainfo.classes)
    head_cfg = cfg.model.pts_bbox_head
    task_class_names = [list(t['class_names']) for t in head_cfg['tasks']]
    # global class id -> (task index, channel index inside the task head)
    cls2task = {}
    for tid, names in enumerate(task_class_names):
        for ch, name in enumerate(names):
            cls2task[class_names.index(name)] = (tid, ch)

    pc_range = cfg.point_cloud_range
    voxel_size = cfg.voxel_size
    out_size_factor = cfg.model.train_cfg.pts.get('out_size_factor', None)
    if out_size_factor is None:
        out_size_factor = head_cfg.get('bbox_coder', {}).get(
            'out_size_factor')
    print(f'pc_range={pc_range}, voxel_size={voxel_size}, '
          f'out_size_factor={out_size_factor}')
    print(f'class_names={class_names}')
    print(f'task_class_names={task_class_names}')

    with open(args.val_info, 'rb') as f:
        val_info = pickle.load(f)
    data_list = val_info['data_list']
    print(f'val samples in info file: {len(data_list)}')

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    assert len(dataset) == len(data_list), \
        f'dataset len {len(dataset)} != info len {len(data_list)}'
    # raw pkl bbox_label uses the converter class order; the dataset
    # remaps it to the training label space (== dataset/head class
    # order) via label_mapping. Apply the same remap here.
    label_mapping = {int(k): int(v) for k, v in dataset.label_mapping.items()}
    print(f'label_mapping (pkl id -> training id): {label_mapping}')

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    captured = {}

    def hook(module, inputs, output):
        # output: List[Dict[str, Tensor]], one dict per task head,
        # output[i]['heatmap']: [B, num_task_classes, H, W] raw logits.
        captured['preds'] = output

    model.pts_bbox_head.register_forward_hook(hook)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pseudo_collate,
        drop_last=False)

    out_preds = {}
    out_gts = {}
    sample_idx = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            results = model.test_step(batch)
            task_preds = captured['preds']
            # CenterHead.forward returns multi_apply(...) output: a tuple
            # with one entry per task, each a list over feature levels
            # (single level here), holding that task's output dict.
            if isinstance(task_preds, tuple):
                task_preds = [t[0] for t in task_preds]
            heatmaps = [t['heatmap'].sigmoid().float().cpu().numpy()
                        for t in task_preds]
            if batch_i == 0:
                print('heatmap shapes:',
                      [h.shape for h in heatmaps])

            for b, ds in enumerate(results):
                info = data_list[sample_idx]
                token = info['token']

                inst = ds.pred_instances_3d
                boxes = inst.bboxes_3d.tensor.cpu().numpy().astype(
                    np.float32)
                scores = inst.scores_3d.cpu().numpy().astype(np.float32)
                labels = inst.labels_3d.cpu().numpy().astype(np.int64)
                out_preds[token] = dict(
                    boxes=boxes, scores=scores, labels=labels)

                gt_rows = []
                for gt in info['instances']:
                    raw_label = int(gt['bbox_label'])
                    if raw_label == -1:
                        continue
                    label = label_mapping[raw_label]
                    bbox = np.asarray(gt['bbox_3d'], dtype=np.float32)
                    x, y, z = bbox[0], bbox[1], bbox[2]
                    sx, sy, sz = bbox[3], bbox[4], bbox[5]
                    volume = float(max(sx * sy * sz, 1e-6))
                    n_pts = int(gt['num_lidar_pts'])
                    lidar_range = float(np.linalg.norm(bbox[:2]))

                    hm_score = None
                    tid, ch = cls2task[label]
                    coor_x = ((x - pc_range[0]) / voxel_size[0]
                              / out_size_factor)
                    coor_y = ((y - pc_range[1]) / voxel_size[1]
                              / out_size_factor)
                    cx, cy = int(coor_x), int(coor_y)
                    hm = heatmaps[tid][b]
                    if 0 <= cx < hm.shape[2] and 0 <= cy < hm.shape[1]:
                        hm_score = float(hm[ch, cy, cx])

                    gt_rows.append(dict(
                        cls_id=label,
                        num_lidar_pts=n_pts,
                        volume=volume,
                        x=float(x), y=float(y), z=float(z),
                        sx=float(sx), sy=float(sy), sz=float(sz),
                        lidar_range=lidar_range,
                        hm_center_score=hm_score))
                out_gts[token] = gt_rows
                sample_idx += 1

            if batch_i % 100 == 0:
                print(f'batch {batch_i}/{len(loader)}, '
                      f'samples done: {sample_idx}')

    assert sample_idx == len(data_list), \
        f'processed {sample_idx} != {len(data_list)}'

    n_pred = sum(len(v['scores']) for v in out_preds.values())
    n_gt = sum(len(v) for v in out_gts.values())
    print(f'done. samples={sample_idx}, preds={n_pred}, gts={n_gt}')

    with open(args.out, 'wb') as f:
        pickle.dump(dict(
            preds=out_preds,
            gts=out_gts,
            class_names=class_names,
            task_class_names=task_class_names,
            pc_range=pc_range,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            checkpoint=args.checkpoint,
        ), f)
    print(f'saved to {args.out}')


if __name__ == '__main__':
    main()
