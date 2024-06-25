# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import os.path as osp
import json
import itertools
from collections import deque
import random

import cv2
import numpy as np
import torch

if __name__ == "__main__":
    import sys
    sys.path.append('/home/msc_lab/zxh/sema-corr/dust3r')

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, is_good_type, transpose_to_landscape
from dust3r.utils.image import imread_cv2
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

# from vis_utils import *
import pandas as pd


class ShapeNet(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, mix_views_by_category_level=False, *args, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.ROOT = ROOT
        self.mask_bg = mask_bg
        self.mix_views_by_category_level = mix_views_by_category_level

        self.load_split_file()
        self.load_scene()
        self._store_obj_idxs_by_category_level()

        # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
        # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
        self.combinations = [(i, j)
                             for i, j in itertools.combinations(range(100), 2)
                             if abs(i-j) % 5 == 0]

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)

    def load_scene(self):
        scene = {}
        catagories = os.listdir(self.ROOT)
        valid_catagory = set(self.valid_synsetId)
        valid_obj = set(self.valid_modelId)
        for catagory in catagories:
            if catagory not in valid_catagory:
                continue
            #### debug on limited catagory only #####
            if not (catagory == '02924116' or catagory == '03261776'):
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
    
    def _store_obj_idxs_by_category_level(self):
        scene_category_map = {}
        scene_list = []
        for k, v in self.scenes.items():
            if not k in scene_category_map:
                scene_category_map[k] = []
            for obj in v:
                scene_list.append((k, obj))
                scene_category_map[k].append(len(scene_list)-1)
        self.scene_list = scene_list
        self.scene_category_map = scene_category_map


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
        # print(obj, instance, imgs_idxs)

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

    def _get_views_by_obj_instance_level(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views
    
    def _mix_views_by_category_level(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # check and enforce the same category
        obj, _ = self.scene_list[idx // len(self.combinations)]
        second_idx = random.choice(self.scene_category_map[obj]) * len(self.combinations) + (idx % len(self.combinations))
        
        views1 = self._get_views_by_obj_instance_level(idx)
        views2 = self._get_views_by_obj_instance_level(second_idx)

        views = [views1[0], views2[1]]
        return views

    def __getitem__(self, idx):
        if self.mix_views_by_category_level:
            return self._mix_views_by_category_level(idx)
        else:
            return self._get_views_by_obj_instance_level(idx)

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = ShapeNet(split='train', ROOT='data/ShapeNet/ShapeNetRendering', cross_mode=True, resolution=224, aug_crop=0)
        
    for idx in np.random.permutation(len(dataset)):
        # views = dataset[idx]
        views = dataset._mix_views_by_category_level(idx)
        assert len(views) == 2
        # print(view_name(views[0]), view_name(views[1]))

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
