from typing import Tuple, Optional, Dict, Any, List
import logging as log
import os
import resource

import torch
from torch.multiprocessing import Pool
import torchvision.transforms
from PIL import Image
import imageio.v3 as iio

from utils.my_tqdm import tqdm

pil2tensor = torchvision.transforms.ToTensor()
# increase ulimit -n (number of open files) otherwise parallel loading might fail
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16192, rlimit[1]))


def _load_video_rhe_livcam(idx: int,
                     liv_paths: List[str],
                     reh_paths: List[str],
                     liv_poses: torch.Tensor,
                     reh_poses: torch.Tensor,
                     out_h: int,
                     out_w: int,
                     load_every: int = 1,
                     liv_mask_paths: List[str] = None,
                     reh_mask_paths: List[str] = None,
                     ):  # -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    filters = [
        ("scale", f"w={out_w}:h={out_h}")
    ]
    reh_all_frames = iio.imread(
        reh_paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,
        filter_sequence=filters,)

    liv_all_frames = iio.imread(
        liv_paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,
        filter_sequence=filters,)
    if reh_mask_paths is not None and liv_mask_paths is not None:
        liv_mask_frames = iio.imread(
            liv_mask_paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,
            filter_sequence=filters,)
        reh_mask_frames = iio.imread(
            reh_mask_paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,
            filter_sequence=filters,)
        all_liv_masks = liv_mask_frames
        reh_mask_frames = reh_mask_frames[...,1:2] #Green channel means human
        liv_mask_frames = liv_mask_frames[...,1:2] 
        all_imgs = zip(liv_all_frames,reh_all_frames,all_liv_masks,liv_mask_frames,reh_mask_frames)
        liv_imgs, reh_imgs, all_mask, liv_mask, reh_mask, timestamps = [], [], [], [], [], []

    else:
        all_imgs = zip(liv_all_frames,reh_all_frames)
        reh_imgs, liv_imgs, timestamps = [], [], []
    for frame_idx, frame in enumerate(all_imgs):
        if frame_idx % load_every != 0:
            continue
        if frame_idx >= 75:  # Only look at the first 10 seconds
            break
        # Frame is np.ndarray in uint8 dtype (H, W, C)
        liv_imgs.append(
            torch.from_numpy(frame[0])
        )
        reh_imgs.append(
            torch.from_numpy(frame[1])
        )
        if reh_mask_paths is not None and liv_mask_paths is not None:
            all_mask.append(
                torch.from_numpy(frame[2])
            )

            liv_mask.append(
                torch.from_numpy(frame[3])
            )
            reh_mask.append(
                torch.from_numpy(frame[4])
            )

        timestamps.append(frame_idx)
    liv_imgs = torch.stack(liv_imgs, 0)
    reh_imgs = torch.stack(reh_imgs, 0)

    #med_img, _ = torch.median(imgs, dim=0)  # [h, w, 3]
    if reh_mask_paths is not None and liv_mask_paths is not None:
        liv_mask = torch.stack(liv_mask, 0)
        reh_mask = torch.stack(reh_mask, 0)
        all_mask = torch.stack(all_mask, 0)

        return (liv_imgs,
            reh_imgs,
            all_mask,
            liv_mask,
            reh_mask,
            liv_poses[idx].expand(len(timestamps), -1, -1),
            reh_poses[idx].expand(len(timestamps), -1, -1),
            torch.tensor(timestamps, dtype=torch.int32))

    else:
        print("errors")

        return (liv_imgs,
            reh_imgs,
            liv_poses[idx].expand(len(timestamps), -1, -1),
            reh_poses[idx].expand(len(timestamps), -1, -1),
            torch.tensor(timestamps, dtype=torch.int32))




def _parallel_loader_kpop_video(args):
    torch.set_num_threads(1)
    return _load_video_rhe_livcam(**args)


def parallel_load_kpop_images(tqdm_title,
                         dset_type: str,
                         num_images: int,
                         **kwargs) -> List[Any]:
    max_threads = 10
    fn = _parallel_loader_kpop_video
    # giac: Can increase to e.g. 10 if loading 4x subsampled images. Otherwise OOM.
    max_threads = 8
    p = Pool(min(max_threads, num_images))

    iterator = p.imap(fn, [{"idx": i, **kwargs} for i in range(num_images)])
    outputs = []
    for _ in tqdm(range(num_images), desc=tqdm_title):
        out = next(iterator)
        if out is not None:
            outputs.append(out)
    return outputs



def _parallel_loader_OF(args):
    torch.set_num_threads(1)
    return _parallel_loader_of_data(**args)

def parallel_load_OF(
                    num_images: int,
                    **kwargs) -> List[Any]:
    max_threads = 10
    fn = _parallel_loader_OF
    # giac: Can increase to e.g. 10 if loading 4x subsampled images. Otherwise OOM.
    max_threads = 8
    p = Pool(min(max_threads, num_images))

    iterator = p.imap(fn, [{"idx": i, **kwargs} for i in range(num_images)])
    outputs = []
    for _ in tqdm(range(num_images)):
        out = next(iterator)
        if out is not None:
            outputs.append(out)
    return outputs

    """    warped_rgb_paths = glob.glob(datadir,'warped','*.mp4')
    warped_conf_paths = glob.glob(datadir,'warped_conf','*.pth')
    sque_conf_paths = glob.glob(datadir,'sque_conf','*.pth')
    sque_flow_paths = glob.glob(datadir,'sque_flow','*.pth')

    """
def _parallel_loader_of_data(idx: int,
                    warped_rgb_paths: List[str],
                    load_every: int = 1,
                    ):  # -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    warped_rgb_frames = iio.imread(
        warped_rgb_paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,)
    rgbs = []
    for frame_idx, frame in enumerate(warped_rgb_frames):
        rgbs.append(
            torch.from_numpy(frame)
        )
    rgbs = torch.stack(rgbs,0)
    return rgbs
    
    
    
    
