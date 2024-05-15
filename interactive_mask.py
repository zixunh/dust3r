# from dust3r_insertion import run_dust3r_insertion
import pyrealsense2 as rs
import os
import sys
import cv2
import json
import argparse
import numpy as np

def initialize_camera():
    # assuming we are using 435i, no need to align the depth and color
    # Create a pipeline
    pipeline = rs.pipeline()
    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    # check whether we are using 515
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
    else:
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

    # config.enable_stream(rs.stream.depth, 640, 480)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    # Get the sensor once at the beginning. (Sensor index: 1)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]


    # Set the exposure anytime during the operation
    sensor.set_option(rs.option.exposure, 156.000)
    sensor.set_option(rs.option.brightness,1)

    if device_product_line == 'L500':
        align = rs.align(rs.stream.color)
    else:
        align = None
    return pipeline, align

def get_rgbd_image(pipeline, align):
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    if align:
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
    else:
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())[..., ::-1]
    # show the image
    # cv2.imshow('RealSense', color_image)
    return depth_image, color_image

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--object', type=str, default = 'diamond', choices=['diamond', 'charger', 'half_circle', 'square', 'tight_peg', 'waterproof'])
    parser.add_argument('--mask_mode', type=str, default = 'interactive_sam', choices=['interactive_sam', 'yolo_sam'])
    parser.add_argument('--scene_dir', type=str, default = '/home/mscfanuc/zixun/dust3r/data/robot_insertion')

    return parser.parse_args()

if __name__ == "__main__":
    opt = parse()
    query_path = os.path.join(opt.scene_dir, opt.object, 'que_views_history')
    if opt.debug:
        pass
    else:
        pipeline, align = initialize_camera()
        # capture the image
        depth_image, color_image = get_rgbd_image(pipeline, align)
        # change the resolution of the depth image to match the color image
        depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        idx = min(len(os.listdir(os.path.join(query_path, 'rgb'))),\
                    len(os.listdir(os.path.join(query_path, 'mask'))))+1
        cv2.imwrite(os.path.join(query_path, 'rgb', str(idx)+'_Color.png'), color_image)

    if opt.debug:
        pass
    elif opt.mask_mode=='yolo_sam':
        cv2.imwrite('/home/mscfanuc/zixun/DTTD_Net/debug/current_frame.jpg', color_image)
        import subprocess
        result = subprocess.run('bash ./detection/demo_yolo_detect.sh', shell=True, capture_output=True, text=True)
        with open('./debug/bbox.json', 'r') as file:
            bbox = json.load(file)['4'][0][0]
            rmin, rmax, cmin, cmax = bbox[1], bbox[3], bbox[0], bbox[2]
        print('[YOLO] bbox detected:', rmin, rmax, cmin, cmax)
        itemid, label = object_detection(color_image, bbox)
        masks = np.transpose(np.repeat(label*255, 3, axis=0),(1,2,0))
        cv2.imwrite('./debug/obj_label.jpg', masks)
        label = label[0,:,:]*itemid
        print(label.shape, np.unique(label))
    elif opt.mask_mode=='interactive_sam':
        from visualization.sam_interactive import *
        init_mode = 'auto'
        obj_name = opt.object
        data_dir = f'./data/robot_insertion/{obj_name}/que_views_history/rgb'
        mask_dir = f'./data/robot_insertion/{obj_name}/que_views_history/mask'
        sam_checkpoint = os.path.join('/home/mscfanuc/zixun/dust3r/visualization/checkpoint', 'sam_vit_b_01ec64.pth')
        model_type = "vit_b"
        device = torch.device('cuda:0')
        mode = 'binary_mask'
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)

        for i, img_path in enumerate(sorted(os.listdir(data_dir))[::-1]):
            mask_name = img_path
            print(img_path)
            rgb_file = os.path.join(data_dir, img_path)
            img = cv2.imread(rgb_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask, init_mode = interactive_mask(img, predictor, init_mode=init_mode)
            save_mask(mask, img_path, mask_dir)
            break
    else:
        raise NotImplementedError

    # import subprocess
    # print('please waiting...')
    # result = subprocess.run(f'/home/mscfanuc/anaconda3/envs/dust3r/bin/python ./dust3r_insertion.py --object {opt.object}', \
    #                         shell=True, capture_output=True, text=True)
    