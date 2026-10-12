import glob
import os
import logging as log
from typing import Tuple, Optional, List

import numpy as np
import torch

from .ray_utils import (
    center_poses, generate_spiral_path, create_meshgrid, stack_camera_dirs, get_rays
)
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset


class LLFFDataset(BaseDataset):
    def __init__(self,
                 datadir,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: int = 4,
                 hold_every: int = 8,
                 contraction: bool = False,
                 ndc: bool = True,
                 near_scaling: float = 0.9,
                 ndc_far: float = 1.0,
                 ):
        if (not contraction) and (not ndc):
            raise ValueError("LLFF dataset expects either contraction or NDC to be enabled.")
        self.downsample = downsample
        self.hold_every = hold_every
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far

        if split == 'render':
            # For rendering path we load all poses, use them to generate spiral poses.
            # No gt images exist!
            assert ndc, "Unable to generate render poses without ndc: don't know near-far."
            image_paths, poses, near_fars, intrinsics = load_llff_poses(
                datadir, downsample=downsample, split='test', hold_every=1,
                near_scaling=self.near_scaling)
            render_poses = generate_spiral_path(
                poses.numpy(), near_fars.numpy(), n_frames=120, n_rots=2, zrate=0.5,
                dt=self.near_scaling)
            self.poses = torch.from_numpy(render_poses).float()
            imgs = None
        else:
            image_paths, self.poses, near_fars, intrinsics = load_llff_poses(
                datadir, downsample=downsample, split=split, hold_every=hold_every,
                near_scaling=self.near_scaling)
            imgs = load_llff_images(image_paths, intrinsics, split)
            imgs = (imgs * 255).to(torch.uint8)
            if split == 'train':
                imgs = imgs.view(-1, imgs.shape[-1])
            else:
                imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])
        num_images = len(self.poses)
        if contraction:
            bbox = torch.tensor([[-2., -2., -2.], [2., 2., 2.]])
            self.near_fars = near_fars
        else:
            bbox = torch.tensor([[-1.5, -1.67, -1.], [1.5, 1.67, 1.]])
            self.near_fars = torch.tensor([[0.0, self.ndc_far]]).repeat(num_images, 1)

        # These are used when contraction=True
        self.global_translation = torch.tensor([0, 0, 1.5])
        self.global_scale = torch.tensor([0.9, 0.9, 1])

        super().__init__(
            datadir=datadir,
            split=split,
            scene_bbox=bbox,
            batch_size=batch_size,
            imgs=imgs,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            is_ndc=ndc,
            is_contracted=contraction,
        )
        """
        log.info(f"LLFFDataset. {contraction=} {ndc=}. Loaded {split} set from {datadir}. "
                 f"{num_images} poses of shape {self.img_h}x{self.img_w}. "
                 f"Images loaded: {imgs is not None}. Near-far[:3]: {self.near_fars[:3]}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")
        """
    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)
            image_id = torch.div(index, h * w, rounding_mode='floor')
            y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
            x = torch.remainder(index, h * w).remainder(w)
            x = x + 0.5
            y = y + 0.5
        else:
            image_id = [index]
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)
        out = {"near_fars": self.near_fars[image_id, :].view(-1, 2)}
        if self.imgs is not None:
            out["imgs"] = self.imgs[index] / 255.0  # (num_rays, 3)   this converts to f32
        else:
            out["imgs"] = None

        c2w = self.poses[image_id]       # (num_rays, 3, 4)
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, True)  # [num_rays, 3]
        rays_o, rays_d = get_rays(camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0,
                                  intrinsics=self.intrinsics, normalize_rd=True)  # h*w, 3
        out["rays_o"] = rays_o
        out["rays_d"] = rays_d
        out["bg_color"] = torch.tensor([[1.0, 1.0, 1.0]])
        return out


def _split_poses_bounds(poses_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    near_fars = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
    intrinsics = Intrinsics(
        width=W, height=H, focal_x=focal, focal_y=focal, center_x=W / 2, center_y=H / 2)
    return poses[:, :, :4], near_fars, intrinsics
def warping_bypose(reh_img, reh_pose, liv_pose):
    reh_img = np.moveaxis(reh_img, 2, 0)
    liv_pose, near_fars, liv_pose_intrinsics = _split_poses_bounds(liv_pose)
    reh_pose, near_fars, reh_pose_intrinsics = _split_poses_bounds(reh_pose)
    liv_pose = np.concatenate([liv_pose[..., 1:2], -liv_pose[..., :1], liv_pose[..., 2:4]], -1)

    reh_pose = np.concatenate([reh_pose[..., 1:2], -reh_pose[..., :1], reh_pose[..., 2:4]], -1)


    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(liv_pose), 1, 1))  # (N_images, 1, 4)
    liv_pose = \
        np.concatenate([liv_pose, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate
    reh_pose = \
        np.concatenate([reh_pose, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    #poses_centered = np.linalg.inv(liv_pose2) @ reh_pose2  # (N_images, 4, 4) #W->W
    rehcam2livcam = liv_pose @ np.linalg.inv(reh_pose)  # (N_images, 4, 4) #C->C

    rehcam2livcam = rehcam2livcam[:, :3]  # (N_images, 3, 4)
    
    #reh_pose, pose_avg = center_poses(reh_pose)
    #liv_pose, pose_avg = center_poses(liv_pose)

    H,W,focal = int(liv_pose_intrinsics.height), int(liv_pose_intrinsics.width), liv_pose_intrinsics.focal_x
    # all positions become x,y,1 (n_rays, 3) these are normalized in H,W
    xx = np.arange(0, W,dtype=float)
    xx=  np.reshape(xx,(1,-1)) 
    xx = np.repeat(xx,H,axis=0)

    yy = np.arange(0, H,dtype=float)
    yy = np.reshape(yy,(-1,1))
    yy = np.repeat(yy,W,axis=1)
    xx = xx.reshape(1 ,H ,W)
    yy = yy.reshape(1 ,H ,W)
    grid = np.concatenate((xx ,yy) ,0)
    another_grid = grid.copy()
    linear_grid = another_grid.reshape(2,-1) 
    linear_grid = np.concatenate((linear_grid,np.ones((1,linear_grid.shape[1],),dtype=np.float32)),0)
    K = np.eye(3)
    K[0, 0] = liv_pose_intrinsics.focal_x 
    K[0, 2] = W/2
    K[1, 1] = liv_pose_intrinsics.focal_y 
    K[1, 2] = H/2 
    inv_reh_k = np.linalg.inv(K)
    cam_grid = inv_reh_k @ linear_grid
    cam_grid = np.stack(
        [
            (linear_grid[0,:] - K[0,2])/K[0,0],
            (linear_grid[1,:] - K[1,2])/K[1,1],\
            np.ones((linear_grid.shape[1]),dtype=np.float32)
        ],-1
        )
    rehcam2livcam = np.squeeze(rehcam2livcam)
    result_coord = cam_grid@rehcam2livcam[:,:3].T  #linear_grid.T @ poses_centered.T 
    result_coord = result_coord.T
    result_coord = np.stack(
        [
            (K[0,0] * result_coord[0,:] + K[0,2]),
            (K[1,1] * result_coord[1,:] + K[1,2]),
        ], 0
        )
    result_coord = np.squeeze(result_coord)

    xx = result_coord[0,:].reshape(1, H, W)#.repeat(1 ,1 ,1 ,1)
    yy = result_coord[1,:].reshape(1, H ,W)#.repeat(1 ,1 ,1 ,1)
    grid = np.concatenate((xx ,yy) ,0)
    grid[0 ,: ,:] = 2.0 *xx.copy() / max( W -1 ,1 ) -1.0
    grid[1 ,: ,:] = 2.0 *yy.copy() / max( H -1 ,1 ) -1.0
    grid =torch.Tensor(grid)
    grid = torch.stack((grid[0 ,: ,:],grid[1 ,: ,:]),2)

    mask = (grid<1.0) * (grid>-1.0)
    
    
    grid =torch.unsqueeze(grid,dim=0)
    

    reh_img =torch.unsqueeze(torch.Tensor(reh_img),dim=0)
    output = F.grid_sample(reh_img,grid, mode='bilinear', align_corners=False)
    output = np.squeeze(output)
    return output, mask


def load_llff_poses_helper(datadir: str, downsample: float, near_scaling: float) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses_bounds = np.load(os.path.join(datadir, 'poses_bounds.npy'))  # (N_images, 17)
    poses, near_fars, intrinsics = _split_poses_bounds(poses_bounds)

    # Step 1: rescale focal length according to training resolution
    intrinsics.scale(1 / downsample)

    # Step 2: correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    # (N_images, 3, 4) exclude H, W, focal
    poses, pose_avg = center_poses(poses)

    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    # See https://github.com/bmild/nerf/issues/34
    near_original = np.min(near_fars)
    scale_factor = near_original * near_scaling  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33
    near_fars /= scale_factor
    poses[..., 3] /= scale_factor

    return poses, near_fars, intrinsics


def load_llff_poses(datadir: str,
                    downsample: float,
                    split: str,
                    hold_every: int,
                    near_scaling: float = 0.75) -> Tuple[
                        List[str], torch.Tensor, torch.Tensor, Intrinsics]:
    int_dsample = int(downsample)
    if int_dsample != downsample or int_dsample not in {4, 8}:
        raise ValueError(f"Cannot downsample LLFF dataset by {downsample}.")

    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    image_paths = sorted(glob.glob(os.path.join(datadir, f'images_{int_dsample}/*')))
    assert poses.shape[0] == len(image_paths), \
        'Mismatch between number of images and number of poses! Please rerun COLMAP!'

    # Take training or test split
    i_test = np.arange(0, poses.shape[0], hold_every)
    img_list = i_test if split != 'train' else list(set(np.arange(len(poses))) - set(i_test))
    img_list = np.asarray(img_list)

    image_paths = [image_paths[i] for i in img_list]
    poses = torch.from_numpy(poses[img_list]).float()
    near_fars = torch.from_numpy(near_fars[img_list]).float()

    return image_paths, poses, near_fars, intrinsics

