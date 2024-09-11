from pckg_util import install_mmcv
install_mmcv()

# standard libraries
import json
import os
import os.path as osp
import pickle as pkl
import pathlib
import random
from abc import ABC
from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm

import logging

# local modules
from innofw.constants import Frameworks, Stages
from innofw.core.datamodules.base import BaseDataModule


ItemInfo = namedtuple('ItemInfo', ['img_path', 'name'])
target_maxes = np.array([[136.7853946685791, 135.2938232421875, 45.29965019226074]])
target_mins = np.array([[-136.7853946685791, -135.2938232421875, -45.29965019226074]])


class Mmdetection3DDataModuleAdapter(BaseDataModule, ABC):
    """
    ...
    ├── data
    │   ├── custom
    │   │   ├── ImageSets  # just names lists, w/o extension or path
    │   │   │   ├── train.txt
    │   │   │   ├── val.txt
    │   │   ├── points
    │   │   │   ├── 000000.bin
    │   │   │   ├── 000001.bin
    │   │   │   ├── ...
    │   │   ├── labels
    │   │   │   ├── 000000.txt
    │   │   │   ├── 000001.txt
    │   │   │   ├── ...
    """
    # Points clouds preprocessing code credit: https://github.com/sadjava
    task = ["3d-detection"]
    framework = [Frameworks.mmdetection]

    def __init__(
            self,
            train: Optional[str],
            test: Optional[str],
            infer: Optional[str],
            num_classes: int,
            val_size: float = 0.2,
            stage=False,
            random_state: int = 0,
            *args,
            **kwargs,
    ):
        super().__init__(train=train, test=test, infer=infer, stage=stage, *args, **kwargs)
        self.train_source, self.test_source, self.infer_source = None, None, None
        if self.train:
            self.train_source = Path(self.train)
        if self.test:
            self.test_source = Path(self.test)
        if self.infer:
            self.infer_source = Path(self.infer)

        self.val_size = val_size
        self.num_classes = num_classes
        self.random_state = random_state

        self.train_set = None
        self.val_set = None
        self.state = {
            'data_path': self.train_source or self.test_source or self.infer_source,
            'save_path': None,
            'train_size': 0.9,
            'center_coords': [True, True, True],  # x, y, z
            'window_size': [100., 100., 20.],
            'train_data_mode': 'full',
            'selectedClasses': ['LEP110_anchor', 'power_lines', 'forest', 'vegetation',
                                'LEP110_prom'],
        }
        self.state['data_path'] = self.state['data_path'].parent
        self.state['save_path'] = osp.join(self.state['data_path'].parent, 'processed_data')

    def get_train_val_sets(self):
        all_items = []
        for p in os.listdir(osp.join(self.state['data_path'], 'pointcloud')):
            all_items.append(
                ItemInfo(name=p, img_path=osp.join(self.state['data_path'], 'pointcloud', p)))
        random.shuffle(all_items)
        train_count = int(len(all_items) * self.state['train_size'])
        train_items = all_items[:train_count]
        val_items = all_items[train_count:]
        return train_items, val_items

    @staticmethod
    def calculate_pcr(items):
        point_cloud_range = [10000, 10000, 10000, -10000, -10000, -10000]
        point_cloud_dim = [0, 0, 0]
        for idx, item in tqdm(enumerate(items), total=len(items), desc='Calculating pcr'):
            pcd = o3d.io.read_point_cloud(item.img_path)
            pcd_np = np.asarray(pcd.points)
            ptc_range = [
                pcd_np[:, 0].min(),
                pcd_np[:, 1].min(),
                pcd_np[:, 2].min(),
                pcd_np[:, 0].max(),
                pcd_np[:, 1].max(),
                pcd_np[:, 2].max()
            ]
            for i in range(3):
                if ptc_range[i] < point_cloud_range[i]:
                    point_cloud_range[i] = ptc_range[i]
                if ptc_range[i + 3] > point_cloud_range[i + 3]:
                    point_cloud_range[i + 3] = ptc_range[i + 3]
                if ptc_range[i + 3] - ptc_range[i] > point_cloud_dim[i]:
                    point_cloud_dim[i] = ptc_range[i + 3] - ptc_range[i]
        print(f'Finished calculating pcr: {point_cloud_range}, pcd: {point_cloud_dim}')
        return point_cloud_range, point_cloud_dim

    def create_splits(self):
        self.train_set, self.val_set = self.get_train_val_sets()
        pcr, pcd = Mmdetection3DDataModuleAdapter.calculate_pcr(self.train_set + self.val_set)
        self.state['point_cloud_range'] = pcr
        self.state['point_cloud_dim'] = pcd

    def centerize_ptc(self, points):
        centerize_vec = [0, 0, 0]
        for i in range(3):
            if self.state['center_coords'][i]:
                dim_trans = points[:, i].min() + (points[:, i].max() - points[:, i].min()) * 0.5
                points[:, i] -= dim_trans
                centerize_vec[i] = -dim_trans

        return points, centerize_vec

    def get_ann_in_framework_format(self, item, sample_idx, pcd_np, pcd_sboxes, sly_ann,
                                    annotations):
        for slide_box_idx, sbox in enumerate(pcd_sboxes):
            ptc_info = {
                'sample_idx': sample_idx,
                'lidar_points': {},
                'instances': {
                    'gt_bboxes_3d': [],
                    'gt_names': [],
                    'gt_labels_3d': [],
                    'box_type_3d': 'LiDAR'
                }
            }
            pcd_eps = 1e-3
            pcd_slide = pcd_np[
                (pcd_np[:, 0] > sbox[0] - pcd_eps) &
                (pcd_np[:, 0] < sbox[1] + pcd_eps) &
                (pcd_np[:, 1] > sbox[2] - pcd_eps) &
                (pcd_np[:, 1] < sbox[3] + pcd_eps) &
                (pcd_np[:, 2] > sbox[4] - pcd_eps) &
                (pcd_np[:, 2] < sbox[5] + pcd_eps)
                ]
            if len(pcd_slide) == 0:
                continue
            slide_name = f"{item.name}_{slide_box_idx}"  # .replace('.', '')
            bin_filename = osp.join("points", f"{slide_name}.bin")
            trans_vec = [0, 0, 0]
            if any(self.state["center_coords"]):
                pcd_slide, trans_vec = self.centerize_ptc(pcd_slide)

            intensity = np.zeros((pcd_slide.shape[0], 1), dtype=np.float32)
            pcd_mins = np.array(self.state['point_cloud_range'][:3])
            pcd_maxes = np.array(self.state['point_cloud_range'][3:])
            pcd_slide = (pcd_slide - pcd_mins) / (pcd_maxes - pcd_mins) * (target_maxes - target_mins) + target_mins
            pcd_slide = np.hstack((pcd_slide, intensity))
            pcd_slide.astype(np.float32).tofile(osp.join(self.state['save_path'], bin_filename))
            ptc_info['lidar_points']['lidar_path'] = bin_filename.split('/')[-1]
            ptc_info['lidar_points']['num_pts_feats'] = 4
            objects2class = {obj['key']: obj['classTitle'] for obj in sly_ann['objects']}
            for fig in sly_ann['figures']:
                if objects2class[fig['objectKey']] not in self.state["selectedClasses"]:
                    continue
                box_info = []  # x, y, z, dx, dy, dz, rot, [vel_x, vel_y]
                pos = fig['geometry']['position']

                if pos['x'] < sbox[0] or pos['x'] >= sbox[1] or \
                        pos['y'] < sbox[2] or pos['y'] >= sbox[3] or \
                        pos['z'] < sbox[4] or pos['z'] >= sbox[5]:
                    continue
                pos_x = pos['x'] + trans_vec[0]
                pos_y = pos['y'] + trans_vec[1]
                pos_z = pos['z'] + trans_vec[2]

                box_pos = [pos_x, pos_y, pos_z]
                box_info.extend(box_pos)
                dim = fig['geometry']['dimensions']
                box_info.extend([dim['x'], dim['y'], dim['z']])
                box_info.extend([fig['geometry']['rotation']['z']])
                # box_info.extend([0, 0])

                ptc_info['instances']['gt_names'].append(objects2class[fig['objectKey']])
                ptc_info['instances']['gt_bboxes_3d'].append(box_info)
                ptc_info['instances']['gt_labels_3d'].append(
                    self.state["selectedClasses"].index(objects2class[fig['objectKey']]))
            ptc_info['instances']['gt_bboxes_3d'] = np.array(ptc_info['instances']['gt_bboxes_3d'],
                                                             dtype=np.float32)
            ptc_info['instances']['gt_bboxes_3d'][:, :3] = (ptc_info['instances']['gt_bboxes_3d'][:, :3] - pcd_mins) / (pcd_maxes - pcd_mins) * (target_maxes - target_mins) + target_mins
            ptc_info['instances']['gt_bboxes_3d'][:, 3:6] = (ptc_info['instances']['gt_bboxes_3d'][:, 3:6] - pcd_mins) / (pcd_maxes - pcd_mins) * (target_maxes - target_mins) + target_mins


            ptc_info['instances']['gt_labels_3d'] = np.array(ptc_info['instances']['gt_labels_3d'],
                                                             dtype=np.int32)
            annotations.append(ptc_info)
        return annotations

    def save_set_to_annotation(self, save_path, items, slide_boxes, subset):
        os.makedirs(osp.dirname(save_path), exist_ok=True)

        annotations = []

        for idx, item in enumerate(items):

            os.makedirs(osp.join(self.state['save_path'], "points"), exist_ok=True)
            pcd = o3d.io.read_point_cloud(item.img_path)
            pcd_np = np.asarray(pcd.points)

            pcd_sboxes = []
            pcdim = self.state["point_cloud_dim"]
            for sbox in slide_boxes:
                pcd_sboxes.append([
                    pcd_np[:, 0].min() + (pcd_np[:, 0].max() - pcd_np[:, 0].min()) * 0.5 - pcdim[
                        0] * 0.5 + sbox[0],
                    pcd_np[:, 0].min() + (pcd_np[:, 0].max() - pcd_np[:, 0].min()) * 0.5 - pcdim[
                        0] * 0.5 + sbox[1],
                    pcd_np[:, 1].min() + (pcd_np[:, 1].max() - pcd_np[:, 1].min()) * 0.5 - pcdim[
                        1] * 0.5 + sbox[2],
                    pcd_np[:, 1].min() + (pcd_np[:, 1].max() - pcd_np[:, 1].min()) * 0.5 - pcdim[
                        1] * 0.5 + sbox[3],
                    pcd_np[:, 2].min() + (pcd_np[:, 2].max() - pcd_np[:, 2].min()) * 0.5 - pcdim[
                        2] * 0.5 + sbox[4],
                    pcd_np[:, 2].min() + (pcd_np[:, 2].max() - pcd_np[:, 2].min()) * 0.5 - pcdim[
                        2] * 0.5 + sbox[5]
                ])

            ann_path = osp.join(self.state['data_path'], "ann", f"{item.name}.json")
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            annotations = self.get_ann_in_framework_format(item, idx, pcd_np, pcd_sboxes, ann,
                                                           annotations)

        os.makedirs(osp.join(osp.dirname(save_path), 'labels'), exist_ok=True)
        all_gt_names = set()
        for ann in annotations:
            name = ann['lidar_points']['lidar_path'][:-4]
            bboxes = ann['instances']['gt_bboxes_3d']
            class_names = ann['instances']['gt_names']
            all_gt_names.update(class_names)
            with open(osp.join(osp.dirname(save_path), 'labels', f"{name}.txt"), 'w') as f:
                for bbox, class_name in zip(bboxes, class_names):
                    print(*bbox, class_name, file=f)

        all_gt_names = list(sorted(all_gt_names))
        os.makedirs(osp.join(osp.dirname(save_path), 'ImageSets'), exist_ok=True)
        with open(osp.join(osp.dirname(save_path), 'ImageSets', f'{subset}.txt'), 'w') as f:
            for ann in annotations:
                print(ann['lidar_points']['lidar_path'][:-4], file=f)

        annotations = {'metainfo': {'categories': {cat: label for cat, label in enumerate(all_gt_names)}},
                       'data_list': annotations}
        with open(save_path, 'wb') as f:
            pkl.dump(annotations, f)

    def get_slide_boxes(self):
        pcd = self.state["point_cloud_dim"].copy()
        ws = self.state["window_size"].copy()
        for i in range(3):
            if pcd[i] < ws[i]:
                ws[i] = pcd[i]

        slides_x, overlap_x = divmod(pcd[0], ws[0])
        slides_y, overlap_y = divmod(pcd[1], ws[1])
        slides_z, overlap_z = divmod(pcd[2], ws[2])
        if overlap_x != 0:
            slides_x += 1
        if overlap_y != 0:
            slides_y += 1
        if overlap_z != 0:
            slides_z += 1

        sboxes = []
        for z in range(int(slides_z)):
            for y in range(int(slides_y)):
                for x in range(int(slides_x)):
                    sboxes.append([
                        0 if x == 0 else ws[0] * x - overlap_x,
                        ws[0] if x == 0 else ws[0] * (x + 1) - overlap_x,
                        0 if y == 0 else ws[1] * y - overlap_y,
                        ws[1] if y == 0 else ws[1] * (y + 1) - overlap_y,
                        0 if z == 0 else ws[2] * z - overlap_z,
                        ws[2] if z == 0 else ws[2] * (z + 1) - overlap_z,
                    ])
        return sboxes, ws

    def prepare_data(self):
        if self.state["train_data_mode"] == 'sliding_window':
            sboxes, self.state["window_size"] = self.get_slide_boxes()
            pcr = [
                -self.state["window_size"][0] * 0.5,
                -self.state["window_size"][1] * 0.5,
                -self.state["window_size"][2] * 0.5,
                self.state["window_size"][0] * 0.5,
                self.state["window_size"][1] * 0.5,
                self.state["window_size"][2] * 0.5
            ]
        elif self.state["train_data_mode"] == 'full':
            sboxes = [[
                0, self.state["point_cloud_dim"][0],
                0, self.state["point_cloud_dim"][1],
                0, self.state["point_cloud_dim"][2]
            ]]
            if any(self.state["center_coords"]):
                pcr = [
                    -self.state["point_cloud_dim"][0] * 0.5 if self.state["center_coords"][0] else
                    self.state["point_cloud_range"][0],
                    -self.state["point_cloud_dim"][1] * 0.5 if self.state["center_coords"][1] else
                    self.state["point_cloud_range"][1],
                    -self.state["point_cloud_dim"][2] * 0.5 if self.state["center_coords"][2] else
                    self.state["point_cloud_range"][2],
                    self.state["point_cloud_dim"][0] * 0.5 if self.state["center_coords"][0] else
                    self.state["point_cloud_range"][3],
                    self.state["point_cloud_dim"][1] * 0.5 if self.state["center_coords"][1] else
                    self.state["point_cloud_range"][4],
                    self.state["point_cloud_dim"][2] * 0.5 if self.state["center_coords"][2] else
                    self.state["point_cloud_range"][5]
                ]
            else:
                pcr = self.state["point_cloud_range"]
        self.state['point_cloud_range'] = pcr

        self.save_set_to_annotation(osp.join(self.state['save_path'], 'custom_infos_train.pkl'), self.train_set,
                                    sboxes, 'train')
        self.save_set_to_annotation(osp.join(self.state['save_path'], 'custom_infos_val.pkl'), self.val_set,
                                    sboxes, 'val')

        logging.info('Finished data preparation')

    def setup_train_test_val(self, **kwargs):
        self.create_splits()
        self.prepare_data()

    def setup_infer(self, **kwargs):
        self.create_splits()
        self.prepare_data()

    def predict_dataloader(self):
        pass

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        pass

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass


if __name__ == '__main__':
    t = Mmdetection3DDataModuleAdapter(data={'source': '/home/karim/workdir/innofw/data/lep3d'},
                                       num_classes=4)
    t.setup_train_test_val()