import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any, List
import pandas as pd
import torch
import torch.utils.data
import numpy as np
from datasets.kpop_dataset_w_of import kpop_dataset

from utils.ema import EMA
from utils.my_tqdm import tqdm
from ops.image import metrics
from ops.image.io import write_video_to_file
from models.lowrank_model import LowrankModel
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss,MotionLoss,BigMotionLoss
)
from tqdm import tqdm as origin_tqdm

class KpopTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 loss_weights: Dict,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 isg_step: int,
                 ist_step: int,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.loss_weight = loss_weights
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        self.ist_step = ist_step
        self.isg_step = isg_step
        self.save_video = save_outputs
        # Switch to compute extra video metrics (FLIP, JOD)
        self.compute_video_metrics = False
        super().__init__(
            loss_weights=loss_weights,
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=False,  # False since we're saving video
            device=device,
            **kwargs)

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs = self.model(
                    rays_o_b, rays_d_b, timestamps=timestamps_d_b, bg_color=bg_color,
                    near_far=near_far)
                    
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k or "brdf" in k:
                        preds[k].append(v.cpu())

        return {k: torch.cat(v, 0) for k, v in preds.items()}
    def render_wo_il_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().render_wo_il_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs = self.model.render_wo_il(
                    rays_o_b, rays_d_b, timestamps=timestamps_d_b, bg_color=bg_color,
                    near_far=near_far)
                    
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k or "brdf" in k:
                        preds[k].append(v.cpu())

        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def render_fig8_step(self,data,time,**kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().render_fig8_step(data,time, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            opt2 = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                light_timestamps = (timestamps_d_b+1.0)/2.0 *(num_frames-1) + time
                light_timestamps = torch.remainder(light_timestamps,num_frames)

                light_timestamps = 2*(light_timestamps/(num_frames-1)) -1.0

                outputs_shift_time_opt2 = self.model.test_shift_time_opt2( 
                    rays_o_b, rays_d_b, obj_timestamps=timestamps_d_b, light_timestamps=light_timestamps, bg_color=bg_color,near_far=near_far                )

                    
                for k, v in outputs_shift_time_opt2.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k:
                        opt2[k].append(v.cpu())

        return    {k: torch.cat(v, 0) for k, v in opt2.items()}
    def render_fig8(self):
        dataset = self.test_dataset
        ill_pred_frames, ill_out_d_frames, ill_brdf_frames, ill_brdf_s_frames, ill_brdf_d_frames = [], [], [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
            if img_idx ==0:
                data['timestamps'] = (32/74)*(torch.ones_like(data['timestamps']))*2 -1.0                
                change_ill = self.render_fig8_step(data,num_frames=self.test_dataset.num_frames, time=-6)
                opt_img, opt_s_frame, opt_d_frame = self.test_shift_time(
                    data, change_ill, dset=dataset, img_idx=img_idx, name=None,
                    save_outputs=self.save_outputs)
                if opt_img is not None:
                    ill_pred_frames.append(opt_img)

        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_fig8.mp4"),
            ill_pred_frames
        )
    def render_frozen_motion(self,time):
        dataset = self.test_dataset
        ill_pred_frames, ill_out_d_frames, ill_brdf_frames, ill_brdf_s_frames, ill_brdf_d_frames = [], [], [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
            if img_idx ==0:
                data['timestamps'] = (32/74)*(torch.ones_like(data['timestamps']))*2 -1.0                
                change_ill = self.render_fig8_step(data,num_frames=self.test_dataset.num_frames, time=-6)
                opt_img, opt_s_frame, opt_d_frame = self.test_shift_time(
                    data, change_ill, dset=dataset, img_idx=img_idx, name=None,
                    save_outputs=self.save_outputs)
                if opt_img is not None:
                    ill_pred_frames.append(opt_img)

        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_fig8.mp4"),
            ill_pred_frames
        )


    
    
    def app_ill(self, data, **kwargs):
        super().app_ill(data, **kwargs)

        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            app_change_ill = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs_change_ill = self.model.render_change_ill( 
                    rays_o_b, rays_d_b,hue_value=-1, il_ratio=0.01, timestamps=timestamps_d_b, bg_color=bg_color,near_far=near_far                )
                for k, v in outputs_change_ill.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k or  "brdf" in k:
                        app_change_ill[k].append(v.cpu())


        return {k: torch.cat(v, 0) for k, v in app_change_ill.items()}
    
    def app_brighter(self, data, **kwargs):
        super().app_brighter(data, **kwargs)

        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            app_change_ill = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs_change_ill = self.model.render_change_ill( 
                    rays_o_b, rays_d_b,hue_value=-1, il_ratio=5.0, timestamps=timestamps_d_b, bg_color=bg_color,near_far=near_far                )
                for k, v in outputs_change_ill.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k or  "brdf" in k:
                        app_change_ill[k].append(v.cpu())


        return {k: torch.cat(v, 0) for k, v in app_change_ill.items()}

    def app_hue(self, data, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().app_hue(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            app_change_hue = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs_change_hue = self.model.render_change_ill( 
                    rays_o_b, rays_d_b,hue_value=0.0, il_ratio=1.0, timestamps=timestamps_d_b, bg_color=bg_color,near_far=near_far                )
                                    
                for k, v in outputs_change_hue.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k or  "brdf" in k:
                        app_change_hue[k].append(v.cpu())
        return  {k: torch.cat(v, 0) for k, v in app_change_hue.items()}
    def supp_test_hue(self, data, hue_value, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().supp_test_hue( data, hue_value, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
    
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            app_change_hue = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs_change_hue = self.model.render_change_ill( 
                    rays_o_b, rays_d_b,hue_value=hue_value, il_ratio=1.0, timestamps=timestamps_d_b, bg_color=bg_color,near_far=near_far                )
                                    
                for k, v in outputs_change_hue.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k or  "brdf" in k:
                        app_change_hue[k].append(v.cpu())
        return  {k: torch.cat(v, 0) for k, v in app_change_hue.items()}
    def supp_change_il_map(self, data, il_map, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().supp_change_il_map( data, il_map, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
    
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            il_map = il_map.to(self.device)

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            app_change_hue = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                il_map_d_b = il_map[b * batch_size: (b + 1) * batch_size].to(self.device)
                outputs_change_hue = self.model.supp_il_map( 
                    rays_o_b, rays_d_b, il_map=il_map_d_b, timestamps=timestamps_d_b, bg_color=bg_color,near_far=near_far                )
                                    
                for k, v in outputs_change_hue.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k or  "brdf" in k:
                        app_change_hue[k].append(v.cpu())
        return  {k: torch.cat(v, 0) for k, v in app_change_hue.items()}

    def app_no_il_and_attention_test(self, data, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().app_no_il_and_attention_test(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            app_change_zero = defaultdict(list)
            test_outputs_attention= defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs_change_zero = self.model.render_change_ill( 
                    rays_o_b, rays_d_b,hue_value=-1, il_ratio=0.0, timestamps=timestamps_d_b, bg_color=bg_color,near_far=near_far                )
                outputs_attention = self.model.test_attention(
                    rays_o_b, rays_d_b, timestamps=timestamps_d_b, bg_color=bg_color,
                    near_far=near_far)
                    
                for k, v in outputs_change_zero.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k or  "brdf" in k:
                        app_change_zero[k].append(v.cpu())
                for k, v in outputs_attention.items():
                    if "out" in k:
                        test_outputs_attention[k].append(v.cpu())

        return  {k: torch.cat(v, 0) for k, v in app_change_zero.items()},\
            {k: torch.cat(v, 0) for k, v in test_outputs_attention.items()}

    def option1_step(self, data, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().option1_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            opt1 = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                light_timestamps = (timestamps_d_b+1.0)/2.0 *(num_frames-1) +20.0
                light_timestamps = torch.remainder(light_timestamps,num_frames)

                light_timestamps = 2*(light_timestamps/(num_frames-1)) -1.0

                outputs_shift_time_opt1 = self.model.test_shift_time_opt1( 
                    rays_o_b, rays_d_b, obj_timestamps=timestamps_d_b, light_timestamps=light_timestamps, bg_color=bg_color,near_far=near_far                )

                    
                for k, v in outputs_shift_time_opt1.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k:
                        opt1[k].append(v.cpu())

        return    {k: torch.cat(v, 0) for k, v in opt1.items()}
    def option2_step(self, data, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().option2_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            opt2 = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                light_timestamps = (timestamps_d_b+1.0)/2.0 *(num_frames-1) + 20.0
                light_timestamps = torch.remainder(light_timestamps,num_frames)

                light_timestamps = 2*(light_timestamps/(num_frames-1)) -1.0

                outputs_shift_time_opt2 = self.model.test_shift_time_opt2( 
                    rays_o_b, rays_d_b, obj_timestamps=timestamps_d_b, light_timestamps=light_timestamps, bg_color=bg_color,near_far=near_far                )

                    
                for k, v in outputs_shift_time_opt2.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k:
                        opt2[k].append(v.cpu())

        return    {k: torch.cat(v, 0) for k, v in opt2.items()}
    def supp_frozen_motion_step(self, time, data, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().supp_frozen_motion_step(time, data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            opt2 = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                object_timestampts = 2.0*(time*torch.ones_like(timestamps_d_b))/(num_frames-1)  -1.0

                outputs_shift_time_opt2 = self.model.test_shift_time_opt2( 
                    rays_o_b, rays_d_b, obj_timestamps=object_timestampts, light_timestamps=timestamps_d_b, bg_color=bg_color,near_far=near_far                )

                    
                for k, v in outputs_shift_time_opt2.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k:
                        opt2[k].append(v.cpu())

        return    {k: torch.cat(v, 0) for k, v in opt2.items()}
    
    def supp_frozen_light_step(self, time, data, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().supp_frozen_light_step(time, data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            opt2 = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                fix_timestampts = 2.0*(time*torch.ones_like(timestamps_d_b))/(num_frames-1)  -1.0

                outputs_shift_time_opt2 = self.model.test_shift_time_opt2( 
                    rays_o_b, rays_d_b, obj_timestamps=timestamps_d_b, light_timestamps=fix_timestampts, bg_color=bg_color,near_far=near_far                )

                    
                for k, v in outputs_shift_time_opt2.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k:
                        opt2[k].append(v.cpu())

        return    {k: torch.cat(v, 0) for k, v in opt2.items()}


    def option3_step(self, data, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().option3_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            opt3 = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                light_timestamps = (timestamps_d_b+1.0)/2.0 *(num_frames-1) +20.0
                light_timestamps = torch.remainder(light_timestamps,num_frames)

                light_timestamps = 2*(light_timestamps/(num_frames-1)) -1.0

                outputs_shift_time_opt3 = self.model.test_shift_time_opt3( 
                    rays_o_b, rays_d_b, obj_timestamps=timestamps_d_b, light_timestamps=light_timestamps, bg_color=bg_color,near_far=near_far                )

                for k, v in outputs_shift_time_opt3.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k:
                        opt3[k].append(v.cpu())

        return    {k: torch.cat(v, 0) for k, v in opt3.items()}
    def option4_step(self, data, **kwargs):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().option4_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]

            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            opt4 = defaultdict(list)
            num_frames = kwargs['num_frames']
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)

                light_timestamps = (timestamps_d_b+1.0)/2.0 *(num_frames-1) + 20.0
                light_timestamps = torch.round(light_timestamps)
                light_timestamps = torch.remainder(light_timestamps,num_frames)

                light_timestamps = 2*(light_timestamps/(num_frames-1)) -1.0

                outputs_shift_time_opt4 = self.model.test_shift_time_opt4( 
                    rays_o_b, rays_d_b, obj_timestamps=timestamps_d_b, light_timestamps=light_timestamps, bg_color=bg_color,near_far=near_far                )

                for k, v in outputs_shift_time_opt4.items():
                    if "rgb" in k or "depth" in k or "illum" in k or "effect" in k:
                        opt4[k].append(v.cpu())

        return    {k: torch.cat(v, 0) for k, v in opt4.items()}
    
    
    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        scale_ok = super().train_step(data, **kwargs)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale_ok

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()


    @torch.no_grad()
    def validate(self):
        dataset = self.test_dataset
        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
        pred_frames, out_d_frames, out_d_only_frames, out_reh_frames, out_depths, brdf_frames, brdf_s_frames, brdf_d_frames, illum_frames, lighting_frames, lighting_w_alpha_frames = [], [], [], [], [], [], [], [], [], [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
    
            preds = self.eval_step(data,num_frames=self.test_dataset.num_frames)
            out_metrics, out_img, out_d_frame, out_d_only_frame, out_reh, out_depth, out_brdf, out_brdf_s, out_brdf_d, out_illum, out_lighting, out_lighting_alpha = self.evaluate_metrics(
                data, preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            pred_frames.append(out_img)
            #Original validation ( View synthesis )
            if out_depth is not None:
                out_depths.append(out_depth)
            if out_reh is not None:
                out_reh_frames.append(out_reh)
            if out_d_frames is not None:
                out_d_frames.append(out_d_frame)
            if out_d_only_frames is not None:
                out_d_only_frames.append(out_d_only_frame)
            if out_brdf is not None:
                brdf_frames.append(out_brdf)
            if out_brdf_s is not None:
                brdf_s_frames.append(out_brdf_s)
            if out_brdf_d is not None:
                brdf_d_frames.append(out_brdf_d)
            if out_illum is not None:
                illum_frames.append(out_illum)
            if out_lighting is not None:
                lighting_frames.append(out_lighting)
            if out_lighting_alpha is not None:
                lighting_w_alpha_frames.append(out_lighting_alpha)

            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        if self.save_video:
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
                pred_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_reh.mp4"),
                out_reh_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                out_depths
            )

            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-d_liv.mp4"),
                out_d_frames
            )

            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-d_only_liv.mp4"),
                out_d_only_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-brdf.mp4"),
                brdf_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-brdf_s.mp4"),
                brdf_s_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-brdf_d.mp4"),
                brdf_d_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-illum.mp4"),
                illum_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-lighting.mp4"),
                lighting_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-lighting_w_alpha.mp4"),
                lighting_w_alpha_frames
            )

        # Calculate JOD (on whole video)
        if self.compute_video_metrics:
            per_scene_metrics["JOD"] = metrics.jod(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )
            per_scene_metrics["FLIP"] = metrics.flip(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )

        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=None),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))
    def render_wo_il_version(self):
        dataset = self.test_dataset
        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
        pred_frames, out_d_frames, out_d_only_frames, out_reh_frames, out_depths, brdf_frames, brdf_s_frames, brdf_d_frames, illum_frames, lighting_frames, lighting_w_alpha_frames = [], [], [], [], [], [], [], [], [], [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
    
            preds = self.render_wo_il_step(data,num_frames=self.test_dataset.num_frames)
            out_metrics, out_img, out_d_frame, out_d_only_frame, out_reh, out_depth, out_brdf, out_brdf_s, out_brdf_d, out_illum, out_lighting, out_lighting_alpha = self.evaluate_metrics(
                data, preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            pred_frames.append(out_img)
            #Original validation ( View synthesis )
            if out_depth is not None:
                out_depths.append(out_depth)
            if out_reh is not None:
                out_reh_frames.append(out_reh)
            if out_d_frames is not None:
                out_d_frames.append(out_d_frame)
            if out_d_only_frames is not None:
                out_d_only_frames.append(out_d_only_frame)
            if out_brdf is not None:
                brdf_frames.append(out_brdf)
            if out_brdf_s is not None:
                brdf_s_frames.append(out_brdf_s)
            if out_brdf_d is not None:
                brdf_d_frames.append(out_brdf_d)
            if out_illum is not None:
                illum_frames.append(out_illum)
            if out_lighting is not None:
                lighting_frames.append(out_lighting)
            if out_lighting_alpha is not None:
                lighting_w_alpha_frames.append(out_lighting_alpha)

            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        if self.save_video:
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_wo_inte.mp4"),
                pred_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_reh_wo_inte.mp4"),
                out_reh_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                out_depths
            )

            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-d_liv_wo_inte.mp4"),
                out_d_frames
            )

            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-d_only_liv_wo_inte.mp4"),
                out_d_only_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-lighting_wo_inte.mp4"),
                lighting_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}-lighting_w_alpha_wo_inte.mp4"),
                lighting_w_alpha_frames
            )

        # Calculate JOD (on whole video)
        if self.compute_video_metrics:
            per_scene_metrics["JOD"] = metrics.jod(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )
            per_scene_metrics["FLIP"] = metrics.flip(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )

        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=None),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    @torch.no_grad()
    def app_test_ill(self):
        dataset = self.test_dataset
        ill_pred_frames, ill_out_d_frames, ill_brdf_frames, ill_brdf_s_frames, ill_brdf_d_frames = [], [], [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
            change_ill = self.app_ill(data,num_frames=self.test_dataset.num_frames)
            ill_img, ill_d_frame, ill_brdf, ill_brdf_s, ill_brdf_d = self.show_applies(
                data, change_ill, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if ill_img is not None:
                ill_pred_frames.append(ill_img)
            if ill_d_frame is not None:
                ill_out_d_frames.append(ill_d_frame)
            if ill_brdf is not None:
                ill_brdf_frames.append(ill_brdf)
            if ill_brdf_s is not None:
                ill_brdf_s_frames.append(ill_brdf_s)
            if ill_brdf_d is not None:
                ill_brdf_d_frames.append(ill_brdf_d)
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_ill.mp4"),
            ill_pred_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}-ill_d_liv.mp4"),
            ill_out_d_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_ill_brdf.mp4"),
            ill_brdf_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_ill_brdf_s.mp4"),
            ill_brdf_s_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_ill_brdf_d.mp4"),
            ill_brdf_d_frames
        )
    @torch.no_grad()
    def app_test_brighter(self):
        dataset = self.test_dataset
        ill_pred_frames, ill_out_d_frames, ill_brdf_frames, ill_brdf_s_frames, ill_brdf_d_frames = [], [], [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
            change_ill = self.app_brighter(data,num_frames=self.test_dataset.num_frames)
            ill_img, ill_d_frame, ill_brdf, ill_brdf_s, ill_brdf_d = self.show_applies(
                data, change_ill, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if ill_img is not None:
                ill_pred_frames.append(ill_img)
            if ill_d_frame is not None:
                ill_out_d_frames.append(ill_d_frame)
            if ill_brdf is not None:
                ill_brdf_frames.append(ill_brdf)
            if ill_brdf_s is not None:
                ill_brdf_s_frames.append(ill_brdf_s)
            if ill_brdf_d is not None:
                ill_brdf_d_frames.append(ill_brdf_d)
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_brighter.mp4"),
            ill_pred_frames
        )

        

    @torch.no_grad()
    def app_test_hue(self):
        dataset = self.test_dataset
        hue_pred_frames, hue_out_d_frames, hue_brdf_frames, hue_brdf_s_frames, hue_brdf_d_frames = [], [], [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
    
            change_hue = self.app_hue(data,num_frames=self.test_dataset.num_frames)
            hue_img, hue_d_frame, hue_brdf, hue_brdf_s, hue_brdf_d = self.show_applies(
                data, change_hue, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if hue_img is not None:
                hue_pred_frames.append(hue_img)
            if hue_d_frame is not None:
                hue_out_d_frames.append(hue_d_frame)
            if hue_brdf is not None:
                hue_brdf_frames.append(hue_brdf)
            if hue_brdf_s is not None:
                hue_brdf_s_frames.append(hue_brdf_s)
            if hue_brdf_d is not None:
                hue_brdf_d_frames.append(hue_brdf_d)
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_hue.mp4"),
            hue_pred_frames
        )

        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}-hue_d_liv.mp4"),
            hue_out_d_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_hue_brdf.mp4"),
            hue_brdf_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_hue_brdf_s.mp4"),
            hue_brdf_s_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_hue_brdf_d.mp4"),
            hue_brdf_d_frames
        )
    @torch.no_grad()
    def supp_ill_map_change(self, il_map,type):
        dataset = self.test_dataset
        hue_pred_frames = []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
            il_map_data = il_map[img_idx,:,:].view(-1,1)
            
            change_hue = self.supp_change_il_map(data,il_map =il_map_data,num_frames=self.test_dataset.num_frames)
            hue_img, hue_d_frame, hue_brdf, hue_brdf_s, hue_brdf_d = self.show_applies(
                data, change_hue, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if hue_img is not None:
                hue_pred_frames.append(hue_img)
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_ill_"+str(type).zfill(3) +".mp4"),
            hue_pred_frames
        )

    @torch.no_grad()
    def supp_multiple_hues(self, hue_value):
        dataset = self.test_dataset
        hue_pred_frames = []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
    
            change_hue = self.supp_test_hue(data,hue_value =hue_value,num_frames=self.test_dataset.num_frames)
            hue_img, hue_d_frame, hue_brdf, hue_brdf_s, hue_brdf_d = self.show_applies(
                data, change_hue, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if hue_img is not None:
                hue_pred_frames.append(hue_img)
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_hue_"+str(hue_value).zfill(3) +".mp4"),
            hue_pred_frames
        )
    @torch.no_grad()
    def supp_frozen_motion(self, motion_time):
        dataset = self.test_dataset
        hue_pred_frames = []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
    
            change_hue = self.supp_frozen_motion_step(time = motion_time,data=data,num_frames=self.test_dataset.num_frames)
            hue_img = self.show_only_rgb(
                data, change_hue, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if hue_img is not None:
                hue_pred_frames.append(hue_img)
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_frozen_motion_"+str(motion_time).zfill(3) +".mp4"),
            hue_pred_frames
        )
    @torch.no_grad()
    def supp_frozen_light(self, motion_time):
        dataset = self.test_dataset
        hue_pred_frames = []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
    
            change_hue = self.supp_frozen_light_step(time = motion_time,data=data,num_frames=self.test_dataset.num_frames)
            hue_img = self.show_only_rgb(
                data, change_hue, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if hue_img is not None:
                hue_pred_frames.append(hue_img)
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_frozen_light_"+str(motion_time).zfill(3) +".mp4"),
            hue_pred_frames
        )
        
    @torch.no_grad()
    def app_test_no_light(self):
        dataset = self.test_dataset
        no_light_preds, no_light_d_frames, no_light_brdf_frames, no_light_brdf_s_frames, no_light_brdf_d_frames = [], [], [], [], []
        zero_values, one_values = [], [] 

        for img_idx, data in origin_tqdm(enumerate(dataset)):
    
            no_light,test_attention = self.app_no_il_and_attention_test(data,num_frames=self.test_dataset.num_frames)
            no_light_img, no_light_d_frame, no_light_brdf, no_light_brdf_s, no_light_brdf_d  = self.show_applies(
                data, no_light, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if img_idx ==0: #because too late..
                zero_value=np.mean(test_attention['zero_out'].cpu().numpy())
                one_value=np.mean(test_attention['one_out'].cpu().numpy())

            #wo light
            if no_light_img is not None:
                no_light_preds.append(no_light_img)
            if no_light_d_frame is not None:
                no_light_d_frames.append(no_light_d_frame)
            if no_light_brdf is not None:
                no_light_brdf_frames.append(no_light_brdf)
            if no_light_brdf_s is not None:
                no_light_brdf_s_frames.append(no_light_brdf_s)
            if no_light_brdf_d is not None:
                no_light_brdf_d_frames.append(no_light_brdf_d)

        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_no_light.mp4"),
            no_light_preds
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}-no_light_d_liv.mp4"),
            no_light_d_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_no_light_brdf.mp4"),
            no_light_brdf_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_no_light_brdf_s.mp4"),
            no_light_brdf_s_frames
        )
        write_video_to_file(
            os.path.join(self.log_dir, f"step{self.global_step}_no_light_brdf_d.mp4"),
            no_light_brdf_d_frames
        )

        with open(os.path.join(self.log_dir, f"step{self.global_step}.txt"), "w") as f:
            f.write("zero:")
            f.write(str(zero_value))
            f.write("ones:")
            f.write(str(one_value))

    @torch.no_grad()
    def opt1_test(self):
        dataset = self.test_dataset
        opt_img_frames, opt_s_frames, opt_d_frames = [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
            opt = self.option1_step(data,num_frames=self.test_dataset.num_frames)
            
            opt_img, opt_s_frame, opt_d_frame = self.test_shift_time(
                data, opt, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)

            #Test shifting time domain
            if opt_img is not None:
                opt_img_frames.append(opt_img)
            if opt_s_frame is not None:
                opt_s_frames.append(opt_s_frame)
            if opt_d_frame is not None:
                opt_d_frames.append(opt_d_frame)
                            
        if self.save_video:


            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt1.mp4"),
                opt_img_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt1_s.mp4"),
                opt_s_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt1_d.mp4"),
                opt_d_frames
            )

    @torch.no_grad()
    def opt2_test(self):
        dataset = self.test_dataset
        opt_img_frames, opt_s_frames, opt_d_frames = [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
            opt = self.option2_step(data,num_frames=self.test_dataset.num_frames)
            
            opt_img, opt_s_frame, opt_d_frame = self.test_shift_time(
                data, opt, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)

            #Test shifting time domain
            if opt_img is not None:
                opt_img_frames.append(opt_img)
            if opt_s_frame is not None:
                opt_s_frames.append(opt_s_frame)
            if opt_d_frame is not None:
                opt_d_frames.append(opt_d_frame)
                            
        if self.save_video:


            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt2.mp4"),
                opt_img_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt2_s.mp4"),
                opt_s_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt2_d.mp4"),
                opt_d_frames
            )
            
    @torch.no_grad()
    def opt3_test(self):
        dataset = self.test_dataset
        opt_img_frames, opt_s_frames, opt_d_frames = [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):

            opt = self.option3_step(data,num_frames=self.test_dataset.num_frames)
            
            opt_img, opt_s_frame, opt_d_frame = self.test_shift_time(
                data, opt, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)

            #Test shifting time domain
            if opt_img is not None:
                opt_img_frames.append(opt_img)
            if opt_s_frame is not None:
                opt_s_frames.append(opt_s_frame)
            if opt_d_frame is not None:
                opt_d_frames.append(opt_d_frame)
                            
        if self.save_video:


            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt3.mp4"),
                opt_img_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt3_s.mp4"),
                opt_s_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt3_d.mp4"),
                opt_d_frames
            )
            
    @torch.no_grad()
    def opt4_test(self):
        dataset = self.test_dataset
        opt_img_frames, opt_s_frames, opt_d_frames = [], [], []
        for img_idx, data in origin_tqdm(enumerate(dataset)):
    
            opt = self.option4_step(data,num_frames=self.test_dataset.num_frames)
            
            opt_img, opt_s_frame, opt_d_frame = self.test_shift_time(
                data, opt, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)

            #Test shifting time domain
            if opt_img is not None:
                opt_img_frames.append(opt_img)
            if opt_s_frame is not None:
                opt_s_frames.append(opt_s_frame)
            if opt_d_frame is not None:
                opt_d_frames.append(opt_d_frame)
                                
        if self.save_video:


            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt4.mp4"),
                opt_img_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt4_s.mp4"),
                opt_s_frames
            )
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}_opt4_d.mp4"),
                opt_d_frames
            )

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)
        if self.train_dataset is not None:
            if -1 < self.isg_step < self.global_step < self.ist_step:
                self.train_dataset.enable_isg()
            elif -1 < self.ist_step < self.global_step:
                self.train_dataset.switch_isg2ist()

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):

        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='s_field'),
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='d_field'),
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='l_field'),

            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='d_field'),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='l_field'),


            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='d_field'),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
            MotionLoss(kwargs.get('motion_loss_weight'),what='d_field'),
            BigMotionLoss(kwargs.get('big_motion_loss_weight'),what='d_field'),

        ]

    @property
    def calc_metrics_every(self):
        return 5


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    log.info(f"Loading Kpopdataset with downsample={data_downsample}")
    tr_dset = kpop_dataset(
        data_dir, split='train', downsample=data_downsample, num_frames = kwargs['num_frames'],
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras', None),
        max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
        isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
        pose_type=kwargs['pose_type'],
        use_mask=kwargs['use_mask'],
        masked_weight=kwargs['masked_weight']

    )
    
    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading

    g = torch.Generator()
    g.manual_seed(0)
    tr_loader = torch.utils.data.DataLoader(
        #tr_dset, batch_size=None, num_workers=2,  #prefetch_factor=4, #pin_memory=True, # initialization error occurs.
        tr_dset, batch_size=None, num_workers=0,
        worker_init_fn=init_dloader_random, generator=g)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, split, **kwargs):
    if 'dnerf' in data_dir:
        downsample = 1.0
    else:
        downsample = 2.0                

    ts_dset = kpop_dataset(
        data_dir, split=split, downsample=downsample, num_frames = kwargs['num_frames'],
        max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
        pose_type=kwargs['pose_type'],
        use_mask=kwargs['use_mask'],
        masked_weight=kwargs['masked_weight']
        )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, application_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if (render_only or application_only) else 'test'
    od.update(init_ts_data(data_dirs[0], split=test_split, **kwargs))
    return od
