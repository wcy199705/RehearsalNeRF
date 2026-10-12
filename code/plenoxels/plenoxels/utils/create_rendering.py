"""Entry point for simple renderings, given a trainer and some poses."""
import os
import logging as log
from typing import Union

import torch

from models.lowrank_model import LowrankModel
from utils.my_tqdm import tqdm
from ops.image.io import write_video_to_file,write_video_to_file_render_only
from runners.kpop_trainer import KpopTrainer

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation



@torch.no_grad()
def render_to_path(trainer:KpopTrainer , extra_name: str = "") -> None:
    """Render all poses in the `test_dataset`, saving them to file
    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    dataset = trainer.test_dataset

    pb = tqdm(total=75, desc=f"Rendering scene")
    frames = []
    decomp_l_imgs = []
    decomp_d_imgs = []
    
    for img_idx, data in enumerate(dataset):
        ts_render = trainer.eval_step(data)

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]
        preds_rgb = (
            ts_render["liv_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .byte()
            .numpy()
        )

        frames.append(preds_rgb)
        pb.update(1)
    pb.close()

    out_fname = os.path.join(trainer.log_dir, f"rendering_path_{extra_name}.mp4")

    write_video_to_file_render_only(out_fname, frames)

    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")

    #out_fname = os.path.join(trainer.log_dir, f"rendering_path_{extra_name}.mp4")
    #write_video_to_file(out_fname, motion_imgs)
    #log.info(f"Saved rendering path with {len(motion_imgs)} frames to {out_fname}")

def rendering_fix_dynamic(trainer: KpopTrainer, dynamic_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = 75#len(dataset)

    frames = []
    for img_idx, data in enumerate(dataset):
        print(img_idx)
        # Linearly interpolated timestamp, normalized between -1, 1
        timestamps = torch.Tensor([img_idx / num_frames]) * 2 - 1
        dynamic_time_tensor = torch.Tensor([dynamic_time / num_frames])  * 2 - 1
        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]


        preds = trainer.rendering_no_inpaint(data=data,human_time=dynamic_time_tensor, light_time = timestamps)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()


        frames.append(
                full_out
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"fix_motion22.mp4")
    print(out_fname)
    write_video_to_file_render_only(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
    
def hue_fix(trainer: KpopTrainer, hue_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = 75#len(dataset)
    frames = []
    for img_idx, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        human_time_stampts = torch.Tensor([img_idx / num_frames]) * 2 - 1
        timestamps = torch.Tensor([hue_time / num_frames])  * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]


        preds = trainer.rendering_changing_hue(data=data,human_time=human_time_stampts, light_time = timestamps)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()


        frames.append(
                full_out
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"human_fix_at_" + str(hue_time).zfill(4)+".mp4")
    write_video_to_file_render_only(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
def custom_single(trainer: KpopTrainer, human_time=0, hue_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = len(dataset)
    frames = []
    for img_idx, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        human_time_stampts = torch.Tensor([human_time / num_frames]) * 2 - 1
        timestamps = torch.Tensor([hue_time / num_frames])  * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]


        preds = trainer.rendering_changing_hue(data=data,human_time=human_time_stampts, light_time = timestamps)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()
        full_out = full_out.clamp(0, 1).mul(255.0).byte().numpy()
        import cv2
        full_out = cv2.cvtColor(full_out,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(trainer.log_dir, f"hue_custom_at_" + str(human_time).zfill(4)+"and_light"+str(hue_time).zfill(3)+'.png'),full_out)#.clamp(0, 1).mul(255.0).byte().numpy())

        return

    out_fname = os.path.join(trainer.log_dir, f"human_fix_at_" + str(hue_time).zfill(4)+".mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")

def hue_change(trainer: KpopTrainer, human_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = 75#len(dataset)
    frames = []
    for img_idx, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        human_time_stampts = torch.Tensor([human_time / num_frames]) * 2 - 1
        timestamps = torch.Tensor([img_idx / num_frames])  * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]


        preds = trainer.rendering_changing_hue(data=data,human_time=human_time_stampts, light_time= timestamps)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()


        frames.append(
                full_out
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"hue_fix_at_" + str(human_time).zfill(4)+".mp4")
    write_video_to_file_render_only(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
def custom_hue(trainer: KpopTrainer, hue, fix_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = 75#len(dataset)
    frames = []
    for img_idx, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        human_time_stampts = torch.Tensor([fix_time / num_frames]) * 2 - 1
        timestamps =  torch.Tensor([img_idx / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]

        
        
        preds = trainer.rendering_custom_hue_vid(data=data,hue=hue, timestamps= timestamps)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()


        frames.append(
                full_out
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"hue_custom2_at_" + str(fix_time).zfill(4)+"and_hue+"+str(hue).zfill(3)+".mp4")
    write_video_to_file_render_only(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
def custom_hue_single_Img(trainer: KpopTrainer, hue, fix_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = len(dataset)
    frames = []
    for _, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        timestamps =  torch.Tensor([fix_time / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[fix_time], dataset.img_w[fix_time]

        
        
        preds = trainer.rendering_custom_hue(data=data,hue=hue,fix_time=timestamps)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()
        full_out = full_out.clamp(0, 1).mul(255.0).byte().numpy()
        import cv2
        full_out = cv2.cvtColor(full_out,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(trainer.log_dir, f"hue_custom_at_" + str(fix_time).zfill(4)+"and_hue"+str(hue).zfill(3)+'.png'),full_out)#.clamp(0, 1).mul(255.0).byte().numpy())
   # out_fname = os.path.join(trainer.log_dir, f"hue_custom_at_" + str(fix_time).zfill(4)+"and_hue+"+str(hue).zfill(3)+".mp4")
    #write_video_to_file(out_fname, frames)
        log.info(f"Saved rendering ")
        return

def no_light(trainer: KpopTrainer):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = len(dataset)
    frames = []
    for img_idx, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        timestamps = torch.Tensor([img_idx / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]


        preds = trainer.no_light(data=data,human_time=timestamps)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()


        frames.append(
                full_out
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"no_light.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")


def rendering_fix_light(trainer: KpopTrainer, light_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = 75#len(dataset)
    frames = []
    for img_idx, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        timestamps = torch.Tensor([img_idx / num_frames]) * 2 - 1
        light_time_tensor = torch.Tensor([light_time / num_frames])  * 2 - 1
        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]


        preds = trainer.rendering_no_inpaint(data=data,human_time=timestamps, light_time = light_time_tensor)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()


        frames.append(
                full_out
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"fix_light.mp4")
    write_video_to_file_render_only(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")


"""
def rendering_fix_dynamic(trainer: KpopTrainer, type='mean', dynamic_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = len(dataset)

    frames = []
    for img_idx, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        timestamps = torch.Tensor([img_idx / num_frames]) * 2 - 1
        dynamic_time = torch.Tensor([dynamic_time / num_frames])  * 2 - 1
        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]


        preds = trainer.rendering_inpainting(data=data,human_time=dynamic_time, light_time = timestamps,reduction_type=type)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()


        frames.append(
                full_out
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"fix_motion.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")


def rendering_fix_light(trainer: KpopTrainer, type='mean', light_time=0):
    # Changyeon Won

    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    camdata = None
    num_frames = len(dataset)
    frames = []
    for img_idx, data in enumerate(dataset):
        # Linearly interpolated timestamp, normalized between -1, 1
        timestamps = torch.Tensor([img_idx / num_frames]) * 2 - 1
        light_time = torch.Tensor([light_time / num_frames])  * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]


        preds = trainer.rendering_inpainting(data=data,human_time=timestamps, light_time = light_time,reduction_type=type)
        full_out = preds["liv_rgb"].reshape(img_h, img_w, 3).cpu()


        frames.append(
                full_out
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"fix_light.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
"""
def normalize_for_disp(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


@torch.no_grad()
def decompose_space_time(trainer: KpopTrainer, extra_name: str = "") -> None:
    """Render space-time decomposition videos for poses in the `test_dataset`.

    The space-only part of the decomposition is obtained by setting the time-planes to 1.
    The time-only part is obtained by simple subtraction of the space-only part from the full
    rendering.

    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    chosen_cam_idx = 15
    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    # Store original parameters from main field and proposal-network field
    parameters = []
    for multires_grids in model.field.grids:
        parameters.append([grid.data for grid in multires_grids])
    pn_parameters = []
    for pn in model.proposal_networks:
        pn_parameters.append([grid_plane.data for grid_plane in pn.grids])

    camdata = None
    for img_idx, data in enumerate(dataset):
        if img_idx == chosen_cam_idx:
            camdata = data
    if camdata is None:
        raise ValueError(f"Cam idx {chosen_cam_idx} invalid.")

    num_frames = img_idx + 1
    frames = []
    for img_idx in tqdm(range(num_frames), desc="Rendering scene with separate space and time components"):
        # Linearly interpolated timestamp, normalized between -1, 1
        camdata["timestamps"] = torch.Tensor([img_idx / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]

        # Full model: turn on time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:
                model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = pn_parameters[i][plane_idx]
        preds = trainer.eval_step(camdata)
        full_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Space-only model: turn off time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:  # time-grids off
                model.field.grids[i][plane_idx].data = torch.ones_like(parameters[i][plane_idx])
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = torch.ones_like(pn_parameters[i][plane_idx])
        preds = trainer.eval_step(camdata)
        spatial_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Temporal model: full - space
        temporal_out = normalize_for_disp(full_out - spatial_out)

        frames.append(
            torch.cat([full_out, spatial_out, temporal_out], dim=1)
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"spacetime_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
