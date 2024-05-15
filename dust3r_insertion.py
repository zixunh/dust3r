from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os
import numpy as np
import torch
import PIL.Image 
# from PIL.Image import Image
# from datareader import *
import tqdm
from matplotlib import pyplot as plt
import torchvision.transforms as tvf
import matplotlib
import argparse

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# from vis_utils import *
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# matplotlib.use('Agg')

def preprocess_img(img, mask, tgt_res):
    # Crop image according to mask and then resize to target resolution, then also resize mask accordingly
    # Keep track of the rescaling factor
    # chcek if mask and img is PIL Image, if so, convert to numpy array
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    if isinstance(mask, PIL.Image.Image):
        mask = np.array(mask)
    mask_bin = (mask.copy()[:, :, None] / 255).astype(np.uint8)
    img = img * mask_bin
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask_idx = np.nonzero(mask)
    min_x, max_x = np.min(mask_idx[1]), np.max(mask_idx[1])
    min_y, max_y = np.min(mask_idx[0]), np.max(mask_idx[0])
    center = np.array([(min_y + max_y) // 2, (min_x + max_x) // 2]) # y, x
    length = int(max(max_x - min_x, max_y - min_y))
    length = length * 1.5 if length < max(mask.shape) else length
    if length % 2 != 0:
        length += 1
    half_length = length / 2
    if center[0] - half_length < 0:
        center[0] = half_length
    if center[0] + half_length > img.shape[0] - 1:
        center[0] = img.shape[0] - half_length
    if center[1] - half_length < 0:
        center[1] = half_length
    if center[1] + half_length > img.shape[1] - 1:
        center[1] = img.shape[1] - half_length
    min_x_, max_x_ = int(center[1] - length / 2), int(center[1] + length / 2)
    min_y_, max_y_ = int(center[0] - length / 2), int(center[0] + length / 2)
    img = img[min_y_:max_y_, min_x_:max_x_, :]
    mask = mask[min_y_:max_y_, min_x_:max_x_]

    # img = cv2.resize(img, tgt_res, interpolation=cv2.INTER_CUBIC)
    # mask = cv2.resize(mask, tgt_res, interpolation=cv2.INTER_NEAREST)
    img = PIL.Image.fromarray(img)
    if mask.max() <= 1:
        mask = mask * 255
    mask = PIL.Image.fromarray(mask)
    # apply a higher resolution to image for featmap resolution alignment?
    img_res = (tgt_res, tgt_res)

    img = img.resize(img_res, PIL.Image.BICUBIC)
    # mask_orig = mask.resize(tgt_res, Image.NEAREST)
    mask = mask.resize(img_res, PIL.Image.NEAREST)
    # img = np.array(img).astype(np.float32) / 255.
    # mask_orig = np.array(mask_orig).astype(np.float32) / 255.
    # mask = np.array(mask).astype(np.float32) / 255.
    spec = np.array([min_x_, min_y_, length]).astype(np.float32)

    return img, mask, spec


def rectify_intrinsic(intrinsic, crop_spec, res):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    # fx, fy, cx, cy = intrinsic_spec['fx'], intrinsic_spec['fy'], intrinsic_spec['cx'], intrinsic_spec['cy']
    # center, length = crop_spec[:2], crop_spec[2]
    min_x, min_y, length = crop_spec[0], crop_spec[1], crop_spec[2]
    # min_y = int(center[0] - length / 2)
    # min_x = int(center[1] - length / 2)
    k = res / length
    cx = cx - min_x
    cy = cy - min_y
    fx = fx * k
    fy = fy * k
    cx = cx * k
    cy = cy * k
    intrinsic = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]).astype(np.float32)
    return intrinsic


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, img_id=None, square_ok=False, verbose=True, mask=False):
    if not mask and img_id is None:
        raise ValueError('img_id must be provided')
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    if mask:
        mask_transform = tvf.ToTensor()
        mask = mask_transform(img)
        return mask
    W2, H2 = img.size
    if verbose:
        print(f' - adding with resolution {W1}x{H1} --> {W2}x{H2}')
    img = dict(img=ImgNorm(img)[None], true_shape=np.int32(
        [img.size[::-1]]), idx=img_id, instance=str(img_id))
    return img

def vis_img(img):
    plt.imshow(img)
    plt.show()

def eval_rel_pose(pose1, pose2):
    translation_distance = np.linalg.norm(pose1[:, 3] - pose2[:, 3]) * 100
    rotation_diff = np.dot(pose1[:, :3], pose2[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return angular_distance, translation_distance

def load_3d(mesh_dir):
    mesh = trimesh.load(mesh_dir)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    return to_origin, bbox


def vis_pose(pose, to_origin, bbox, reader, i, show=False):
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                        is_input_rgb=True)
    if show:
        plt.imshow(vis)
        plt.show()
    else:
        os.makedirs(f'{reader.video_dir}/track_vis_dust3r', exist_ok=True)
        imageio.imwrite(f'{reader.video_dir}/track_vis_dust3r/{reader.id_strs[i]}.png', vis)


def analysis_plot(ang_deviation, trans_deviation, ang_err, trans_error):
    # x is the frame number, y is the deviation, cm or degree
    plt.plot(ang_deviation, label='angular deviation')
    plt.plot(ang_err, label='angular error')
    plt.xlabel('frame number')
    plt.ylabel('degree')
    plt.legend()
    plt.show()
    plt.plot(trans_deviation, label='translational deviation')
    plt.plot(trans_error, label='translational error')
    plt.xlabel('frame number')
    plt.ylabel('cm')
    plt.legend()
    plt.show()


def rectify_conf(output, mask_ref=None, mask_query=None):
    if mask_query is not None:
        mask_ref.append(mask_query)
    else:
        mask_ref.append(torch.ones_like(mask_ref[0]))
    mask_ref = torch.cat(mask_ref, dim=0)
    mask_ref1 = mask_ref[output['view1']['idx']]
    mask_ref2 = mask_ref[output['view2']['idx']]
    # mask2 = torch.cat((mask_ref, mask_query), dim=0)
    # mask1 = torch.cat((mask_query, mask_ref), dim=0)
    output['pred1']['conf'][mask_ref1 == 0] = 1
    output['pred2']['conf'][mask_ref2 == 0] = 1
    return output

def rectify_conf_test(output, mask_ref=None):
    mask_ref = torch.cat(mask_ref, dim=0)
    mask_ref1 = mask_ref[output['view1']['idx']]
    mask_ref2 = mask_ref[output['view2']['idx']]
    # mask2 = torch.cat((mask_ref, mask_query), dim=0)
    # mask1 = torch.cat((mask_query, mask_ref), dim=0)
    output['pred1']['conf'][mask_ref1 == 0] = 1
    output['pred2']['conf'][mask_ref2 == 0] = 1
    return output


def intrinsic_matrix(params):
    # Unpack the parameters
    fx, fy, cx, cy = params

    # Create the intrinsic matrix
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    return intrinsic_matrix

def load_ref_img(ref_path, num, pose_available=True, intr_available=True, preprocess=True, intr=None, cam_pose=None, load_orig_pose=False):
    ref_imgs = []
    ref_poses = []
    ref_intrs = []
    ref_masks = []
    orig_pose = []
    rgb_dir = os.path.join(ref_path, 'rgb')
    rgb_files = sorted(os.listdir(rgb_dir))[::-1]
    mask_dir = os.path.join(ref_path, 'mask')
    mask_files = sorted(os.listdir(mask_dir))[::-1]
    pose_dir = os.path.join(ref_path, 'pose') if pose_available else None
    pose_files = sorted(os.listdir(pose_dir))[::-1] if pose_available else []
    intr_dir = os.path.join(ref_path, 'intr') if intr_available else None
    intr_files = sorted(os.listdir(intr_dir))[::-1] if intr_available else []
    pose_idx = [True] * num + [False]
    for i in range(num):
        rgb_path = os.path.join(rgb_dir, rgb_files[i])
        mask_path = os.path.join(mask_dir, mask_files[i])
        print(f"captured:{rgb_path},{mask_path}")
        if i == len(pose_files):
            pose_available = False
        if pose_available:
            pose_path = os.path.join(pose_dir, pose_files[i])
            if pose_path.endswith('.npy'):
                pose = np.load(pose_path)
            elif pose_path.endswith('.txt'):
                pose = np.loadtxt(pose_path)
            if load_orig_pose:
                orig_pose.append(pose)
            if cam_pose is not None:
                pose = np.linalg.inv(cam_pose) @ pose
            pose = np.linalg.inv(pose)
        else:
            pose = None
        if intr_available:
            intr_path = os.path.join(intr_dir, intr_files[i])
            intr = np.loadtxt(intr_path)
            intr[1, 2] = intr[1, 2] - (512 - 384) / 2
        rgb = PIL.Image.open(rgb_path)
        mask = PIL.Image.open(mask_path)
        if preprocess:
            rgb, mask, spec = preprocess_img(rgb, mask, tgt_res=512)
            # vis_img(rgb)
            if intr_available or intr is not None:
                intr_rec = rectify_intrinsic(intr.copy(), spec, 512)
            else:
                intr_rec = None
        rgb = resize_img(rgb, size=512, img_id=i)
        mask = resize_img(mask, size=512, mask=True)
        # intr = rectify_intrinsic(intr, spec, 512)

        ref_imgs.append(rgb)
        ref_poses.append(pose)
        ref_intrs.append(intr_rec)
        ref_masks.append(mask)
    if load_orig_pose:
        return ref_imgs, ref_poses, ref_intrs, ref_masks, pose_idx, orig_pose
    else:
        return ref_imgs, ref_poses, ref_intrs, ref_masks, pose_idx


def compute_score(output, num_ref, topk=1):
    topk_ = topk
    topk = min(num_ref,topk)
    if topk_ != topk: print('Alert! The value of topk is modified.')

    conf1 = output['pred1']['conf'].clone()
    conf2 = output['pred2']['conf'].clone()
    conf1 = conf1.reshape(conf1.shape[0], -1).mean(dim=-1)
    conf2 = conf2.reshape(conf2.shape[0], -1).mean(dim=-1)
    score = conf1 * conf2
    # print(score.shape, conf1.shape, conf2.shape)
    # print(score)
    assert score.shape[0] == num_ref*(num_ref+1) and num_ref>=1
    half_length = int(score.shape[0] / 2)
    # idx_ignore = [i for i in range(0,int(num_ref*(num_ref-1)/2))] + [i for i in range(half_length, num_ref**2)]
    # for i in idx_ignore: score[i] += float('-inf')
    score1 = score[int(num_ref*(num_ref-1)/2):half_length].unsqueeze(0)
    score2 = score[num_ref**2:].unsqueeze(0)
    score = torch.cat((score1, score2), dim=0).mean(dim=0)
    print('query involved scores:', score)
    score_im_idx = torch.argsort(score, descending=True)
    max_ref_idx = score_im_idx[:topk].numpy()

    ref_idx0 = max_ref_idx.min()
    ref_idx1 = max_ref_idx.max()
    que_idx = num_ref
    print('selected topk ref:', ref_idx0, ref_idx1)

    max_pair_idx1 = int(num_ref*(num_ref-1)/2) + max_ref_idx
    max_pair_idx2 = max_pair_idx1 + half_length
    pair_idx0 = [int(ref_idx1*(ref_idx1-1)/2)+ref_idx0, \
                 int(ref_idx1*(ref_idx1-1)/2)+ref_idx0+half_length] \
                if num_ref>=2 else []
    idx = np.concatenate([pair_idx0, max_pair_idx1, max_pair_idx2])  
    print(idx)
    return list(idx), ref_idx0,ref_idx1,que_idx


def nearest_pair(output, pair_idx):
    output_nearest = output.copy()
    view1 = output_nearest['view1']
    view1['img'] = view1['img'][pair_idx]
    view1['true_shape'] = view1['true_shape'][pair_idx]
    view1['idx'] = [1, 2, 2, 0, 0, 1]
    view1['instance'] = ['1', '2', '2', '0', '0', '1']
    view2 = output_nearest['view2']
    view2['img'] = view2['img'][pair_idx]
    view2['true_shape'] = view2['true_shape'][pair_idx]
    view2['idx'] = [0, 0, 1, 1, 2, 2]
    view2['instance'] = ['0', '0', '1', '1', '2', '2']
    pred1 = output_nearest['pred1']
    pred1['pts3d'] = pred1['pts3d'][pair_idx]
    pred1['conf'] = pred1['conf'][pair_idx]
    pred2 = output_nearest['pred2']
    pred2['pts3d_in_other_view'] = pred2['pts3d_in_other_view'][pair_idx]
    pred2['conf'] = pred2['conf'][pair_idx]
    return output_nearest



def run_dust3r_insertion(
    query_path,
    obj_name = 'diamond',
    num_ref = 8,
    model_path = "/home/mscfanuc/zixun/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    scene_dir = '/home/mscfanuc/zixun/dust3r/data/robot_insertion'
):
    ref_path = os.path.join(scene_dir, obj_name, f'ref_views_{num_ref}_gripper')

    # init
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 5000
    res = 512
    topk = 2

    intr = np.load(os.path.join(scene_dir, 'camera_intrinsics.npz'))['arr_0.npy']
    cam_pose_file = np.load(os.path.join(scene_dir, 'camera_pose_new.npz'))
    cam_pose = np.concatenate((cam_pose_file['camera_R'], cam_pose_file['camera_t']), axis=1)
    cam_pose = np.concatenate((cam_pose, np.array((0, 0, 0, 1))[None, :]))
    intr = intrinsic_matrix(intr)
    ref_imgs, ref_poses, ref_intrs, ref_masks, ref_idx, orig_pose = load_ref_img(\
        ref_path, num_ref, pose_available=True, intr_available=False, intr=intr, cam_pose=cam_pose, load_orig_pose=True)
    model = load_model(model_path, device)

    query_imgs, query_poses, query_intrs, query_masks, query_idx = load_ref_img(query_path, 1, pose_available=False, intr_available=False, intr=intr, load_orig_pose=False)
    query_imgs[0]['idx'] = num_ref
    query_imgs[0]['instance'] = str(num_ref)
    imgs = ref_imgs + query_imgs
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    print(output.keys(), output['view1']['idx'], output['view2']['idx'])
    pair_idx, ref_idx0,ref_idx1,que_idx = compute_score(output, num_ref, topk=topk)
    output = rectify_conf(output, mask_ref=ref_masks, mask_query=query_masks[0])
    output = nearest_pair(output, pair_idx)
    # vis_3D(output)
    print(output.keys(), output['view1']['idx'], output['view2']['idx'])
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    scene.preset_intrinsics(ref_intrs + query_intrs)
    print('len(ref_poses)', len(ref_poses))
    scene.preset_pose([ref_poses[ref_idx0],ref_poses[ref_idx1]], [True] * topk + [False])
    # scene.preset_pose(ref_poses, [True] * num_ref + [False])
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    # scene.show()
    poses = scene.get_im_poses()

    tgt_pose_0 = poses[0].detach().cpu().numpy()
    query_pose = poses[-1].detach().cpu().numpy()
    real_pose = orig_pose @ np.linalg.inv(tgt_pose_0) @ query_pose
    print(query_pose)
    return query_pose

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--object', type=str, default = 'diamond', choices=['diamond', 'charger', 'half_circle', 'square', 'tight_peg', 'waterproof'])
    parser.add_argument('--scene_dir', type=str, default = '/home/mscfanuc/zixun/dust3r/data/robot_insertion')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse()
    query_path = os.path.join(opt.scene_dir, opt.object, 'que_views_history')
    query_pose = run_dust3r_insertion(query_path, opt.object, 8)
    idx = len(os.listdir(os.path.join(query_path,'rgb')))
    np.save(os.path.join(query_path,'pose',f'pose_{idx}.npy'),query_pose)
