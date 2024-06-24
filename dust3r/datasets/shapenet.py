# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np

if __name__ == "__main__":
    import sys
    sys.path.append('/home/msc_lab/zxh/sema-corr/dust3r/')

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
# from vis_utils import *
import pandas as pd


class ShapeNet(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.ROOT = ROOT
        self.mask_bg = mask_bg

        self.load_split_file()
        self.load_scene()

        scene_list = []
        for k, v in self.scenes.items():
            for obj in v:
                scene_list.append((k, obj))
        self.scene_list = scene_list

        # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
        # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
        self.combinations = [(i, j)
                             for i, j in itertools.combinations(range(100), 2)
                             if abs(i-j) % 5 == 0]

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = np.arange(100)
        im1_idx, im2_idx = self.combinations[idx % len(self.combinations)]

        # add a bit of randomness
        last = len(image_pool)-1

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            view_idx = im_idx

            impath = osp.join(self.ROOT, obj, instance, 'rgb', f'{view_idx:04n}.png')

            # load camera params
            pose_path = osp.join(self.ROOT, obj, instance, 'pose', f'{view_idx:04n}.txt')
            camera_pose = np.loadtxt(pose_path).astype(np.float32)
            intr_path = osp.join(self.ROOT, obj, instance, 'intrinsics.npy')
            intrinsics = np.load(intr_path).astype(np.float32)

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthpath = osp.join(self.ROOT, obj, instance, 'depth', f'{view_idx:04n}.exr')
            depthmap = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)[:, :, 0]
            maskmap = (depthmap < 10000).astype(np.uint8)
            depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(65535)


            if mask_bg:
                # update the depthmap with mask
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='ShapeNet',
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1],
            ))
        return views


    def load_scene(self):
        scene = {}
        catagories = os.listdir(self.ROOT)
        valid_catagory = set(self.valid_synsetId)
        valid_obj = set(self.valid_modelId)
        for catagory in catagories:
            if catagory not in valid_catagory:
                continue
            #### debug on one catagory only #####
            if catagory != '02924116':
                continue
            scene[catagory] = []
            objects = os.listdir(os.path.join(self.ROOT, catagory))
            for obj in objects:
                if obj not in valid_obj:
                    continue
                scene[catagory].append(obj)
        self.scenes = scene

    def load_split_file(self):
        file_dir = osp.join(osp.dirname(self.ROOT), 'all.csv')
        dtype = {
            'synsetId': 'string',
            'modelId': 'string',
            'split': 'string',
        }
        usecols = ['synsetId', 'modelId', 'split']
        df = pd.read_csv(file_dir, dtype=dtype, usecols=usecols)
        data_split = df['split'].to_numpy()
        synsetId = df['synsetId'].to_numpy()
        modelId = df['modelId'].to_numpy()
        split_idx = data_split == self.split
        self.valid_synsetId = synsetId[split_idx]
        self.valid_modelId = modelId[split_idx]
        


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = ShapeNet(split='train', ROOT='data/ShapeNet/ShapeNetRendering', resolution=224, aug_crop=0)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
        # viz.rend(file_name='test.png')
        # viz.web_show()
        # break
