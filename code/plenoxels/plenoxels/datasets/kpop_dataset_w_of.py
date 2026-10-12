import glob
import json
import logging as log
import math
import os
import time
from collections import defaultdict
from typing import Optional, List, Tuple, Any, Dict
import numpy as np
import torch
import gc
from .base_dataset import BaseDataset
from .data_loading import parallel_load_kpop_images, parallel_load_OF
from .intrinsics import Intrinsics
from .llff_dataset import load_llff_poses_helper,warping_bypose
from .ray_utils import (
    generate_spherical_poses, create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path, stack_camera_dirs_kpop,get_rays_kpop,generate_specific_view
)

class kpop_dataset(BaseDataset):
    #Read Reh data/ Liv data
    
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 pose_type: str,
                 split: str,
                 num_frames: int,
                 use_mask:bool,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 masked_weight: bool = True,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 k: int = 5,
                 ndc_far: float = 2.6):
        self.keyframes = keyframes
        self.k = k
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = False
        self.masked_weight = masked_weight
        self.ist = False
        self.use_mask = use_mask
        self.only_human = False


        self.num_frames = num_frames
        # self.lookup_time = False
        self.per_cam_near_fars = None
        self.global_translation = torch.tensor([0, 0, 0])
        self.global_scale = torch.tensor([1, 1, 1])
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far
        self.median_imgs = None
        if contraction and ndc:
            raise ValueError("Options 'contraction' and 'ndc' are exclusive.")
        if "lego" in datadir or "dnerf" in datadir:
            dset_type = "synthetic"
        else:
            dset_type = "llff"

        # Note: timestamps are stored normalized between -1, 1.
        if dset_type == "llff":
            
            if split == "render":
                
                print("split?",split)
                assert ndc, "Unable to generate render poses without ndc: don't know near-far."
                pre_cam_poses_liv, pre_cam_poses_reh,per_cam_near_fars, intrinsics, _,_  = load_kpopvideo_poses(
                    datadir, pose_type=pose_type, downsample=self.downsample, split='all', near_scaling=self.near_scaling)
                render_poses = generate_spiral_path(
                    pre_cam_poses_liv.numpy(), per_cam_near_fars.numpy(), n_frames=self.num_frames,
                    n_rots=2.0, zrate=0.5, dt=self.near_scaling, percentile=60)
                self.liv_poses = torch.from_numpy(render_poses).float()
                self.reh_poses = torch.from_numpy(render_poses).float()
                self.per_cam_near_fars = torch.tensor([[0.4, self.ndc_far]]) 
                timestamps = torch.linspace(0, self.num_frames -1, len(self.liv_poses))
            
                """
                if split =="render":
                    print("split?",split)
                    assert ndc, "Unable to generate render poses without ndc: don't know near-far."
                    pre_cam_poses_liv, pre_cam_poses_reh,per_cam_near_fars, intrinsics, _,_  = load_kpopvideo_poses(
                        datadir, pose_type=pose_type, downsample=self.downsample, split='all', near_scaling=self.near_scaling)
                    
                    render_poses = generate_specific_view(
                        pre_cam_poses_liv.numpy(), per_cam_near_fars.numpy(), n_frames=self.num_frames,
                        angle=1.2, zrate=1.0, dt=self.near_scaling, percentile=60) 
                    
                    render_poses = generate_specific_view(
                        pre_cam_poses_liv.numpy(), per_cam_near_fars.numpy(), n_frames=self.num_frames,
                        angle=0.0, radius=2.0, zrate=3.0, dt=self.near_scaling, percentile=60) 
                    self.liv_poses = torch.from_numpy(render_poses).float()
                    self.reh_poses = torch.from_numpy(render_poses).float()
                    self.per_cam_near_fars = torch.tensor([[0.4, self.ndc_far]]) 
                    timestamps = torch.linspace(0.0, self.num_frames-1, len(self.liv_poses))
                """
            else:
                if use_mask:
                    pre_cam_poses_liv, pre_cam_poses_reh,per_cam_near_fars, intrinsics, liv_videopaths, reh_videopaths, liv_mask_paths, reh_mask_paths = load_kpopvideo_poses_w_mask(
                        datadir, pose_type=pose_type, downsample=self.downsample, split=split, near_scaling=self.near_scaling)
                else:
                    pre_cam_poses_liv, pre_cam_poses_reh,per_cam_near_fars, intrinsics, liv_videopaths, reh_videopaths  = load_kpopvideo_poses(
                        datadir, pose_type=pose_type, downsample=self.downsample, split=split, near_scaling=self.near_scaling)
                if split == 'test':
                    keyframes = False
                if use_mask:
                    liv_poses, reh_poses, liv_imgs, reh_imgs, all_masks, liv_masks, reh_masks, timestamps = load_kpopvideo_data(
                        liv_paths=liv_videopaths,reh_paths=reh_videopaths,liv_cam_poses=pre_cam_poses_liv,reh_cam_poses=pre_cam_poses_reh, intrinsics=intrinsics,
                        split=split, liv_mask_paths=liv_mask_paths, reh_mask_paths=reh_mask_paths)
                else:
                    liv_poses, reh_poses, liv_imgs, reh_imgs, timestamps = load_kpopvideo_data(
                        liv_paths=liv_videopaths,reh_paths=reh_videopaths,liv_cam_poses=pre_cam_poses_liv,reh_cam_poses=pre_cam_poses_reh, intrinsics=intrinsics,
                        split=split)
                if split == 'train':
                    warped_rgb, warped_conf, sque_conf, sque_flow = load_kpop_OF(
                    datadir = datadir, downsample=downsample
                    )
                    self.warped_rgb = warped_rgb.view(-1,warped_rgb.shape[-1])
                    self.warped_conf = warped_conf.view(-1,warped_conf.shape[-1])
                    self.sque_conf = sque_conf.view(-1,sque_conf.shape[-2],sque_conf.shape[-1])
                    self.sque_flow = sque_flow.view(-1,sque_flow.shape[-2],sque_flow.shape[-1])
                    self.interval = sque_conf.shape[-2]//2
                    self.add_temps = []
                    for interval in range(-self.interval,self.interval+1):
                        if interval !=0:
                            self.add_temps.append(interval*torch.ones((batch_size))/(num_frames-1))
                    self.add_temps = torch.stack(self.add_temps,dim=-1)
                    del warped_conf
                    del warped_rgb
                    del sque_conf
                    del sque_flow
                else:
                    self.warped_rgb = None
                    self.warped_conf = None
                    self.sque_conf = None
                    self.sque_flow = None
                #read mask in v_mask folders. (For human annotations.)                  
                if self.masked_weight and self.use_mask:
                    self.mask = all_masks.view(-1, all_masks.shape[-1])

                self.liv_poses = liv_poses.float()
                self.reh_poses = reh_poses.float()

                if contraction:
                    self.per_cam_near_fars = per_cam_near_fars.float()
                else:
                    self.per_cam_near_fars = torch.tensor(
                        [[0.0, self.ndc_far]]).repeat(per_cam_near_fars.shape[0], 1)
            # These values are tuned for the salmon video
            self.global_translation = torch.tensor([0, 0, 2.]) 
            self.global_scale = torch.tensor([0.5, 0.6, 1])
            # Normalize timestamps between -1, 1
            timestamps = (timestamps.float() / (self.num_frames-1)) * 2 - 1
        else:
            raise ValueError(datadir)

        self.timestamps = timestamps
        if split == 'train':
            self.timestamps = self.timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        assert self.timestamps.min() >= -1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."
        if split != 'render':
            if liv_imgs is not None and liv_imgs.dtype != torch.uint8:
                liv_imgs = (liv_imgs * 255).to(torch.uint8)
            if reh_imgs is not None and reh_imgs.dtype != torch.uint8:
                reh_imgs = (reh_imgs * 255).to(torch.uint8)
        else:
            liv_imgs = None
            reh_imgs = None
            self.liv_imgs = None
            self.reh_imgs = None
            self.liv_masks = None
            self.reh_masks = None
        if split == 'train': # order: 
            self.liv_imgs = liv_imgs.view(-1, liv_imgs.shape[-1]) #shape: 1350,760,1352,3
            self.reh_imgs = reh_imgs.view(-1, reh_imgs.shape[-1])
            if self.use_mask:
                self.liv_masks = liv_masks.view(-1, liv_masks.shape[-1]).cuda()
                self.reh_masks = reh_masks.view(-1, reh_masks.shape[-1]).cuda()
        elif reh_imgs is not None and  liv_imgs is not None:
            self.liv_imgs = liv_imgs.view(-1, intrinsics.height * intrinsics.width, liv_imgs.shape[-1])
            self.reh_imgs = reh_imgs.view(-1, intrinsics.height * intrinsics.width, reh_imgs.shape[-1])
            if self.use_mask:
                self.liv_masks = liv_masks.view(-1, intrinsics.height * intrinsics.width, liv_masks.shape[-1])
                self.reh_masks = reh_masks.view(-1, intrinsics.height * intrinsics.width, reh_masks.shape[-1])
        

        #https://stackoverflow.com/questions/45811421/python-create-image-with-new-camera-position
        # ISG/IST weights are computed on 4x subsampled data.
        weights_subsampled = 1#int(4 / downsample)
        if scene_bbox is not None:
            scene_bbox = torch.tensor(scene_bbox)
        else:
            scene_bbox = get_bbox(datadir, is_contracted=contraction, dset_type=dset_type)
        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            liv_imgs=self.liv_imgs,
            reh_imgs=self.reh_imgs,
            liv_masks = self.liv_masks,
            reh_masks = self.reh_masks,
            sampling_weights=None,  # Start without importance sampling, by default
            weights_subsampled=weights_subsampled,
        )
        self.isg_weights = None
        self.train_rp = False
        self.ist_weights = None
        if not os.path.exists(os.path.join(datadir, f"temporal_hues.pt")):
            #self.hues = torch.load(os.path.join(datadir, f"temporal_hues.pt"))
            #print("hues loaded")
            t_s = time.time()
            self.hues = get_hues(self.liv_imgs.view(-1, intrinsics.height, intrinsics.width, self.liv_imgs.shape[-1])
                                    , self.reh_imgs.view(-1, intrinsics.height, intrinsics.width, self.reh_imgs.shape[-1])
                                    , self.liv_masks.view(-1, intrinsics.height, intrinsics.width, self.liv_masks.shape[-1])
                                    , self.reh_masks.view(-1, intrinsics.height, intrinsics.width, self.reh_masks.shape[-1])
                                    , num_frames=self.num_frames,k=self.k)
            torch.save(self.hues, os.path.join(datadir, f"temporal_hues.pt"))
            t_e = time.time()
            log.info(f"Computed {self.hues.shape[0]} hues in {t_e - t_s:.2f}s.")

            #self.hues = torch.unsqueeze(self.hues,dim=0).repeat(self.liv_poses.shape[0]//self.num_frames,1,1).reshape(-1,self.hues.shape[-1])
            #self.hues = self.hues[:, None, None, :].repeat(                                                          1, intrinsics.height, intrinsics.width, 1).reshape(-1,self.hues.shape[-1])  # [n_frames* h * w, k]
        if self.masked_weight:
            if split == "train" and dset_type == 'llff':  # Only use importance sampling with DyNeRF videos
                if os.path.exists(os.path.join(datadir, f"ism_weights.pt")):
                    
                    self.ism_weights = torch.load(os.path.join(datadir, f"ism_weights.pt"))
                    log.info(f"Reloaded {self.ism_weights.shape[0]} ISM weights from file.")
                else:
                    self.set_ism_from_video()

                    # Precompute ISG weights
                    t_s = time.time()
                    #self.isg_weights = (self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                    torch.save(self.ism_weights, os.path.join(datadir, f"ism_weights.pt"))
                    t_e = time.time()
                    log.info(f"Computed {self.ism_weights.shape[0]} ISM weights in {t_e - t_s:.2f}s.")
                self.enable_ism()
        
        gc.collect()

        """
        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.liv_poses)} images of size {self.img_h}x{self.img_w}. "
                 f"Images loaded: {self.liv_imgs is not None}. "
                 f"Masks loaded: {self.liv_masks is not None}. "
                 f"{len(torch.unique(timestamps))} timestamps. Near-far: {self.per_cam_near_fars}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")
                 
        """

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")
    def set_ism_from_video(self, lms_coef=(1, 1, 1)):
            m_idx = (self.mask[:,1] >128) 
            l_idx = (self.mask[:,0] >128)* (self.mask[:,1]<=128)
            s_idx = m_idx+l_idx#(self.mask[:,0] <=128) + (self.mask[:,1] <=128) #torch.where((l_idx==0 ) * (m_idx==0),1,0) 
            s_idx = s_idx<1.0
            l = torch.sum(l_idx)
            m = torch.sum(m_idx)
            s = torch.sum(s_idx)
            
            l_idx = l_idx.nonzero()
            m_idx = m_idx.nonzero()
            s_idx = s_idx.nonzero()
            
            log.info(f"light : {l}")
            log.info(f"motion : {m}")
            log.info(f"static : {s}")
            log.info(f"total : {l+m+s}")
            if not self.mask.size()[0] == l+m+s and not self.train_rp:
                raise ValueError("not match mask size and total")
                
            self.ism_weights = torch.ones(self.mask.size()[0], dtype=torch.float32)
            print(self.ism_weights.size(), torch.sum(self.ism_weights))
            print(torch.min(self.ism_weights), torch.max(self.ism_weights))
            
            total = l+m+s
            if not l == 0:
                self.ism_weights[l_idx] = self.ism_weights[l_idx]/(1+l/total) * lms_coef[0]    
            if not m == 0:
                self.ism_weights[m_idx] = self.ism_weights[m_idx]/(1+m/total) * lms_coef[1]
            if not s == 0:
                self.ism_weights[s_idx] = self.ism_weights[s_idx]/(1+s/total) * lms_coef[2]
                
            if self.train_rp:
                not_s_idx = torch.logical_not(self.mask[:,2] > 0)
                log.info(f"not {torch.sum(not_s_idx)}")
                not_s_idx = not_s_idx.nonzero()

                self.ism_weights[not_s_idx] = 0
        
                
            ism_sum = torch.sum(self.ism_weights)
            print(self.ism_weights.size(), ism_sum, torch.mean(self.ism_weights))
            print(torch.min(self.ism_weights), torch.max(self.ism_weights))        
            if not ism_sum == 1:
                self.ism_weights = self.ism_weights / ism_sum    
                
    def enable_ism(self):
        self.ism = True
        self.isg = False
        self.ist = False
        self.sampling_weights = self.ism_weights
        log.info("Enabled ISM weights.")


    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)  # [batch_size // (weights_subsampled**2)]
            if  self.sampling_weights is None:
                # Nothing special to do, either weights_subsampled = 1, or not using weights.
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            else:
                #print("sampling!")


                # We must deal with the fact that ISG/IST weights are computed on a dataset with
                # different 'downsampling' factor. E.g. if the weights were computed on 4x
                # downsampled data and the current dataset is 2x downsampled, `weights_subsampled`
                # will be 4 / 2 = 2.
                # Split each subsampled index into its 16 components in 2D.
                #hsub, wsub = h // self.weights_subsampled, w // self.weights_subsampled
                hsub, wsub = h,w
                image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
                ysub = torch.remainder(index, hsub * wsub).div(wsub, rounding_mode='floor')
                xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
                # xsub, ysub is the first point in the 4x4 square of finely sampled points
                x, y = [], []
                for ah in range(1):#self.weights_subsampled):
                    for aw in range(1):#self.weights_subsampled):
                        x.append(xsub + aw)#x.append(xsub * self.weights_subsampled + aw)
                        y.append(ysub + ah)
                        #y.append(ysub * self.weights_subsampled + ah)
                x = torch.cat(x)
                y = torch.cat(y)
                image_id = image_id.repeat(self.weights_subsampled ** 2)
                # Inverse of the process to get x, y from index. image_id stays the same.
                index = x + y * w + image_id * h * w

            x, y = x + 0.5, y + 0.5
        else:
            image_id = [index]
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)
        out = {
            #"hues": self.hues[index,:],
            "timestamps": self.timestamps[index],      # (num_rays or 1, )
            "liv_imgs": None,
            "reh_imgs": None,
            "liv_masks": None,
            "reh_masks": None,
            "warped_rgb": None,
            "warped_conf": None,
            "sque_rays_o":None,
            "sque_rays_d":None,
            "sque_conf":None,
            "sampled_times":None,
        }
        if self.split == 'train':
            num_frames_per_camera = len(self.liv_imgs) // (len(self.per_cam_near_fars) * h * w)
            camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)
            out['near_fars'] = self.per_cam_near_fars[camera_id, :]
        else:
            out['near_fars'] = self.per_cam_near_fars  # Only one test camera

        if self.liv_imgs is not None:
            out['liv_imgs'] = (self.liv_imgs[index] / 255.0).view(-1, self.liv_imgs.shape[-1])
        if self.reh_imgs is not None:
            out['reh_imgs'] = (self.reh_imgs[index] / 255.0).view(-1, self.reh_imgs.shape[-1])
        if self.reh_masks is not None:
            out['reh_masks'] = (self.reh_masks[index]>128).float().view(-1, self.reh_masks.shape[-1])
        if self.liv_masks is not None:
            out['liv_masks'] = (self.liv_masks[index]>128).float().view(-1, self.liv_masks.shape[-1])
       #warped_rgb, warped_conf, sque_conf, sque_flow
        if self.split == 'train':
            if self.warped_rgb is not None:
                out['warped_rgb'] = (self.warped_rgb[index] / 255.0).view(-1, self.warped_rgb.shape[-1])
            if self.warped_conf is not None:
                out['warped_conf'] = (self.warped_conf[index]).view(-1, self.warped_conf.shape[-1])
            if self.sque_conf is not None:
                out['sque_conf'] = (self.sque_conf[index]).view(-1, self.sque_conf.shape[-2],self.sque_conf.shape[-1])

        liv_c2w = self.liv_poses[image_id]                                    # [num_rays or 1, 3, 4]
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, True)  # [num_rays, 3]
        out['rays_o'], out['rays_d'] = get_rays(
            camera_dirs, liv_c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics,
            normalize_rd=True) # [num_rays, 3]
        if self.split == 'train':
            if self.sque_flow is not None:
                sque_flow = (self.sque_flow[index]).view(-1, self.sque_flow.shape[-2],self.sque_flow.shape[-1])
                moved_x = x[...,None]  + sque_flow[...,0]
                moved_y = y[...,None] + sque_flow[...,1]
                camera_dirs_kpop = stack_camera_dirs_kpop(moved_x, moved_y, self.intrinsics, True)  # [num_rays, 4, 3]
                out['sque_rays_o'], out['sque_rays_d'] = get_rays_kpop(
                    camera_dirs_kpop, liv_c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics,
                    normalize_rd=True) # [num_rays, 4, 3]
                temp = self.timestamps[index][:,None] + self.add_temps
                out['sampled_times'] = torch.minimum(torch.maximum(temp,-torch.ones_like(temp)),torch.ones_like(temp))
                            
        liv_imgs = out['liv_imgs']
        reh_imgs = out['reh_imgs']
        # Decide BG color
        bg_color = torch.ones((1, 3), dtype=torch.float16, device=dev)
        #bg_color = None
        if self.split == 'train' and liv_imgs.shape[-1] == 4:
            bg_color = torch.rand((1, 3), dtype=torch.float16, device=dev)
        out['bg_color'] = bg_color
        # Alpha compositing
        if liv_imgs is not None and liv_imgs.shape[-1] == 4:
            liv_imgs = liv_imgs[:, :3] * liv_imgs[:, 3:] + bg_color * (1.0 - liv_imgs[:, 3:])
        if reh_imgs is not None and reh_imgs.shape[-1] == 4:
            reh_imgs = reh_imgs[:, :3] * reh_imgs[:, 3:] + bg_color * (1.0 - reh_imgs[:, 3:])
        out['liv_imgs'] = liv_imgs
        out['reh_imgs'] = reh_imgs
        return out


def get_bbox(datadir: str, dset_type: str, is_contracted=False) -> torch.Tensor:
    """Returns a default bounding box based on the dataset type, and contraction state.

    Args:
        datadir (str): Directory where data is stored
        dset_type (str): A string defining dataset type (e.g. synthetic, llff)
        is_contracted (bool): Whether the dataset will use contraction

    Returns:
        Tensor: 3x2 bounding box tensor
    """
    if is_contracted:
        radius = 2
    elif dset_type == 'synthetic':
        radius = 1.5
    elif dset_type == 'llff':
        return torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])




def load_kpopvideo_poses_w_mask(datadir: str,
                         pose_type:str,
                         downsample: float,
                         split: str,
                         near_scaling: float) -> Tuple[
                            torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information (root)
        pose_type (str) : type of pose (reh,liv,all)
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each liv_camera's data.
        List[str]: List of length N containing the path to each reh_camera's data.
        
    """
    
    pose_dir = os.path.join(datadir,'colmap_' + pose_type, 'ws')    
    poses, near_fars, intrinsics = load_llff_poses_helper(pose_dir, downsample, near_scaling)
    reh_videopaths =  np.array(glob.glob(os.path.join(datadir, 'reh','videos', '*.mp4')))  # [n_cameras]
    liv_videopaths =  np.array(glob.glob(os.path.join(datadir, 'liv','videos', '*.mp4')))  # [n_cameras]
    liv_maskpaths = np.array(glob.glob(os.path.join(datadir, 'liv','v_mask', '*.mp4')))  # [n_cameras]
    reh_maskpaths = np.array(glob.glob(os.path.join(datadir, 'reh','v_mask', '*.mp4')))  # [n_cameras]

    if pose_type == 'all':
        assert poses.shape[0] == (len(liv_videopaths)+len(reh_videopaths)), \
            'Mismatch between number of cameras and number of poses!'
    else:
        assert poses.shape[0] == len(liv_videopaths), \
            'Mismatch between number of cameras and number of poses!'
    if pose_type == 'all':
        liv_poses = poses[:poses.shape[0]//2,:]
        reh_poses = poses[poses.shape[0]//2:,:]
    else:
        liv_poses = poses
        reh_poses = poses
    reh_videopaths.sort()
    liv_videopaths.sort()
    reh_maskpaths.sort()
    liv_maskpaths.sort()

    # The first camera is reserved for testing
    if split == 'train':
        split_ids = np.arange(1, liv_poses.shape[0])
    elif split == 'test':
        split_ids = np.array([0])
    else:
        split_ids = np.arange(liv_poses.shape[0])
    liv_poses = torch.from_numpy(liv_poses[split_ids])
    reh_poses = torch.from_numpy(reh_poses[split_ids])

    near_fars = torch.from_numpy(near_fars[split_ids])
    liv_videopaths = liv_videopaths[split_ids].tolist()
    reh_videopaths = reh_videopaths[split_ids].tolist()
    liv_maskpaths = liv_maskpaths[split_ids].tolist()
    reh_maskpaths = reh_maskpaths[split_ids].tolist()



    return liv_poses, reh_poses, near_fars, intrinsics, liv_videopaths, reh_videopaths, liv_maskpaths, reh_maskpaths
def load_kpopvideo_poses(datadir:str,
                         pose_type:str,
                         downsample: float,
                         split: str,
                         near_scaling: float) -> Tuple[
                            torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information (root)
        pose_type (str) : type of pose (reh,liv,all)
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each liv_camera's data.
        List[str]: List of length N containing the path to each reh_camera's data.
        
    """ 
    
    pose_dir = os.path.join(datadir,'colmap_' + pose_type, 'ws')    
    poses, near_fars, intrinsics = load_llff_poses_helper(pose_dir, downsample, near_scaling)
    reh_videopaths =  np.array(glob.glob(os.path.join(datadir, 'reh','videos', '*.mp4')))  # [n_cameras]
    liv_videopaths =  np.array(glob.glob(os.path.join(datadir, 'liv','videos', '*.mp4')))  # [n_cameras]

    if pose_type == 'all':
        assert poses.shape[0] == (len(liv_videopaths)+len(reh_videopaths)), \
            'Mismatch between number of cameras and number of poses!'
    else:
        assert poses.shape[0] == len(liv_videopaths), \
            'Mismatch between number of cameras and number of poses!'
    if pose_type == 'all':
        liv_poses = poses.shape[:poses.shape[0]//2,:]
        reh_poses = poses.shape[poses.shape[0]//2:,:]
    else:
        liv_poses = poses
        reh_poses = poses
    reh_videopaths.sort()
    liv_videopaths.sort()
    
    # The first camera is reserved for testing
    if split == 'train':
        split_ids = np.arange(1, poses.shape[0])
    elif split == 'test':
        split_ids = np.array([0])
    else:
        split_ids = np.arange(poses.shape[0])

    liv_poses = torch.from_numpy(liv_poses[split_ids])
    reh_poses = torch.from_numpy(reh_poses[split_ids])

    near_fars = torch.from_numpy(near_fars[split_ids])
    liv_videopaths = liv_videopaths[split_ids].tolist()
    reh_videopaths = reh_videopaths[split_ids].tolist()



    return liv_poses, reh_poses, near_fars, intrinsics, liv_videopaths, reh_videopaths






def load_kpopvideo_data(liv_paths: List[str],
                        reh_paths: List[str],
                        liv_cam_poses: torch.Tensor,
                        reh_cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        #keyframes: bool,
                        #keyframes_take_each: Optional[int] = None,
                        liv_mask_paths: List[str] = None,
                        reh_mask_paths: List[str] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
    #    raise ValueError(f"'keyframes_take_each' must be a positive number, "
    #                     f"but is {keyframes_take_each}.")

    loaded = parallel_load_kpop_images(
        dset_type="video",
        tqdm_title=f"Loading {split} data",
        num_images=len(liv_paths),
        liv_paths=liv_paths,
        reh_paths=reh_paths,
        liv_mask_paths = liv_mask_paths,
        reh_mask_paths = reh_mask_paths,
        liv_poses=liv_cam_poses,
        reh_poses=reh_cam_poses,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        #load_every=keyframes_take_each if keyframes else 1,
    )
    if liv_mask_paths is not None and reh_mask_paths is not None:
        liv_imgs, reh_imgs, all_masks,liv_masks, reh_masks, liv_poses, reh_poses, timestamps = zip(*loaded)
        liv_masks = torch.cat(liv_masks, 0)              # [N, h, w]
        reh_masks = torch.cat(reh_masks, 0)              # [N, h, w]
        all_masks = torch.cat(all_masks, 0)

    else:
        liv_imgs, reh_imgs, liv_poses, reh_poses, timestamps = zip(*loaded)
    # Stack everything together
    timestamps = torch.cat(timestamps, 0)  # [N]
    liv_poses = torch.cat(liv_poses, 0)            # [N, 3, 4]
    reh_poses = torch.cat(reh_poses, 0)            # [N, 3, 4]

    liv_imgs = torch.cat(liv_imgs, 0)              # [N, h, w, 3]
    reh_imgs = torch.cat(reh_imgs, 0)              # [N, h, w, 3]

    if liv_mask_paths is not None and reh_mask_paths is not None:
    
        return liv_poses, reh_poses, liv_imgs, reh_imgs, all_masks, liv_masks, reh_masks, timestamps
    else:
        return liv_poses, reh_poses, liv_imgs, reh_imgs, timestamps



#warped_rgb, warped_conf, sque_conf, sque_flow = 
def load_kpop_OF( datadir, downsample ):
    """

    Args:
        datadir (_type_, optional): _description_. Defaults to datadir.
        downsample (_type_, optional): _description_. Defaults to downsample.

    Returns:
        _type_: _description_
    """
    if downsample != 2.0:
        print("error(downsampling is not 2!")
        
    warped_rgb_paths = glob.glob(os.path.join(datadir,'warped','*.mp4'))
    warped_rgb_paths.sort()
    rgbs = parallel_load_OF(
        num_images = len(warped_rgb_paths),
        warped_rgb_paths = warped_rgb_paths,
    )
    rgbs = torch.stack(rgbs,0)
    split_ids = np.arange(1, len(rgbs))
    rgbs = rgbs[split_ids]
    warped_confs = torch.load(os.path.join(datadir,'warped_conf.pth'))[split_ids].half() #,map_location=lambda storage, loc: storage.cuda(2))[split_ids].half() #2.7GB
    print("Loaded_warped_confs!")
    sque_confs = torch.load(os.path.join(datadir,'sque_conf.pth'))[split_ids].half() #map_location=lambda storage, loc: storage.cuda(2))[split_ids].half() 
    print("Loaded_sque_confs!")
    sque_flow = torch.load(os.path.join(datadir,'sque_flow.pth'))[split_ids].half() 
    print("Loaded_sque_flow!")

    return rgbs, warped_confs, sque_confs, sque_flow


@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # noqa
    )  # [num_cameras, num_frames, h, w, 3]
    # differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    # squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


def rgb2hue(img):
    #img : [P,C]
    if img.dtype != np.float32:
        img = img/255

    r = img[...,0]
    g = img[...,1]
    b = img[...,2]
    max_ch = np.argmax(img,axis=-1)
    max_val = np.max(img,axis=-1)
    min_val = np.min(img,axis=-1)
    delta = max_val - min_val
    no_delta = np.where(delta==0)
    max_red_idxs = np.where(max_ch==0)
    max_green_idxs = np.where(max_ch==1)
    max_blue_idxs = np.where(max_ch==2)
    
    hue = np.zeros_like(r)

    hue[max_red_idxs] = (g[max_red_idxs]-b[max_red_idxs])/delta[max_red_idxs]
    hue[max_green_idxs] = 2.0 + (b[max_green_idxs]-r[max_green_idxs])/delta[max_green_idxs]
    hue[max_blue_idxs] = 4.0 + (r[max_blue_idxs]-g[max_blue_idxs])/delta[max_blue_idxs]
    hue[no_delta] = 0.0
    hue = hue * 60.0
    hue[hue<0] = hue[hue<0] + 360.0
    hue[hue==360] = 0.0
    hue = np.round(hue)
    return hue

from sklearn.cluster import KMeans
from tqdm import trange
@torch.no_grad()
def get_hues(liv_imgs:torch.Tensor, reh_imgs:torch.Tensor, liv_masks:torch.Tensor,reh_masks:torch.Tensor, num_frames:int, k:int):
    # imgs is [num_cameras * num_frames, h, w, 3]

    total_frame, h,w,c = liv_imgs.shape

    num_cameras = total_frame//num_frames
    hues = np.zeros((num_frames,k),dtype=np.float32)

    total_liv_imgs = liv_imgs.view(num_cameras,num_frames, -1, c).float().div_(255.0).cpu().numpy()
    total_reh_imgs = reh_imgs.view(num_cameras,num_frames, -1, c).float().div_(255.0).cpu().numpy()
    total_liv_masks = liv_masks.view(num_cameras,num_frames, -1, 1).float().div_(255.0).cpu().numpy()
    total_reh_masks = reh_masks.view(num_cameras,num_frames, -1, 1).float().div_(255.0).cpu().numpy()
    total_liv_masks = total_liv_masks < 0.5
    total_reh_masks = total_reh_masks < 0.5
    total_masks = total_liv_masks * total_reh_masks
    diff_maps = total_masks * np.abs(total_liv_imgs - total_reh_imgs)
    weight_map = np.sum(diff_maps,axis=-1) > 1.0
    ###check gkrl
    for time in trange(num_frames):
        weight = weight_map[:,time,:]
        diff_map = diff_maps[:,time,:,:]
        idxs = np.where(weight)
        diff_map_lin = diff_map[idxs[0], idxs[1], :]
        diff_map_lin = rgb2hue(diff_map_lin)
        selected_color = np.expand_dims(diff_map_lin,axis=-1)
        model = KMeans(n_clusters = k, random_state=30)
        model.fit(selected_color)
        predict_labels = model.fit_predict(selected_color)
        total = predict_labels.shape[0]
        max_num = 0
        for i in range(0,k):
            if max_num < len(predict_labels[predict_labels==i]):
                indxes_i = np.where(predict_labels==i)
                color_i = np.squeeze(selected_color[indxes_i,:],axis=0)
                max_num = len(predict_labels[predict_labels==i])
                max_color = np.median(color_i,axis=0,keepdims=False)
        for i in range(0,k):
            indxes_i = np.where(predict_labels==i)
            color_i = np.squeeze(selected_color[indxes_i,:],axis=0)
            mean_color_i = np.median(color_i,axis=0,keepdims=False)
            if indxes_i[0].shape[0]/total < 0.05:
                mean_color_i = max_color
            hues[time,i] = mean_color_i/360 
    hues = torch.Tensor(hues)    
    return hues
                
    

