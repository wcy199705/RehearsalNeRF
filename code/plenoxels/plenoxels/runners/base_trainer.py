import abc
import random
import logging
import math
import os
from copy import copy
from typing import Iterable, Optional, Union, Dict, Tuple, Sequence, MutableMapping

log = logging.getLogger(__name__)
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import gc
from utils.timer import CudaTimer
from utils.ema import EMA
from models.lowrank_model import LowrankModel
from utils.my_tqdm import tqdm
from ops.image import metrics
from ops.image.io import write_png
from runners.regularization import Regularizer
from ops.lr_scheduling import (
    get_cosine_schedule_with_warmup, get_step_schedule_with_warmup
)
def masked_loss(mask,est):
    if len(est.shape) == 1:
        est = torch.unsqueeze(est,dim=-1)
    if torch.sum(mask*est) > (3*torch.sum(mask)):
        log.warning("masked_loss: sum(mask*est) exceeds 3*sum(mask)")
    loss =  (torch.sum(mask*est)/(torch.sum(mask) +1e-8))/est.shape[-1]
    if not torch.isfinite(loss).all():
        log.warning("masked_loss: non-finite loss detected")
    return  loss

def negative_loss(value,ones):
    return 1/(ones + torch.pow(value,2))
def custom_bceloss(output, target):
    eps = torch.full(output.shape,fill_value=1e-8,device=output.device)
    tensor = -(target*torch.log(output + eps) + (1-target)*torch.log(1-output + eps))
    
    return torch.mean(tensor)

def rgb2hue(img):
    #img : [P,C]
    r = img[...,0]
    g = img[...,1]
    b = img[...,2]
    max_ch = torch.argmax(img,axis=-1)

    max_val = torch.max(img,axis=-1).values
    min_val = torch.min(img,axis=-1).values
    delta = max_val - min_val
    no_delta = torch.where(delta==0)
    max_red_idxs = torch.where(max_ch==0)
    max_green_idxs = torch.where(max_ch==1)
    max_blue_idxs = torch.where(max_ch==2)
    
    hue = torch.zeros_like(r)

    hue[max_red_idxs] = (g[max_red_idxs]-b[max_red_idxs])/delta[max_red_idxs]
    hue[max_green_idxs] = 2.0 + (b[max_green_idxs]-r[max_green_idxs])/delta[max_green_idxs]
    hue[max_blue_idxs] = 4.0 + (r[max_blue_idxs]-g[max_blue_idxs])/delta[max_blue_idxs]
    hue[no_delta] = 0.0
    hue = hue * 60.0
    hue[hue<0] = hue[hue<0] + 360.0
    hue = torch.round(hue)
    return hue
def reh_liv_diff_loss(est,data,ones):

    loss_weight = (ones - data['liv_masks'])*(ones - data['reh_masks'])*torch.pow(data['liv_imgs'] - data['reh_imgs'],2)
    loss = masked_loss(loss_weight, torch.pow(data['liv_imgs'] - est['liv_rgb'],2))
    
    return loss
def illum_loss_fn(est,data,ones):
    loss = masked_loss((ones - data['liv_masks'])*(ones - data['reh_masks']),
                       (torch.mean(torch.pow(data['liv_imgs']-data['reh_imgs'] -est['diff_rgb'],2 ),dim=-1)))
    
    return loss

def reh_illum_error(est,data,ones):
    loss_weight = (ones - data['liv_masks'])*(ones - data['reh_masks'])*torch.pow(data['liv_imgs'] - data['reh_imgs'],2)

    loss = masked_loss(loss_weight * torch.pow(data['reh_imgs'] - est['reh_rgb'],2))
    return loss

def il_d_weight_loss(est,data,ones):
    loss_weight = (ones - data['liv_masks'])*(ones - data['reh_masks'])*torch.pow(data['liv_imgs'] - data['reh_imgs'],2)
    loss = masked_loss(loss_weight, est['weight_d'])
    return loss
def Mc_loss(x,k):
    return 1/torch.sqrt(torch.pow(x,k)+1)
def hue_loss(data, est):

    hue_time = est['hues'][:,0,:]
    est_hue = rgb2hue(est['liv_d_rgb'])
    hue_diff = torch.minimum(-torch.abs(hue_time - est_hue[:,None]) + 360, torch.abs(hue_time - est_hue[:,None]))
    hue_diff  = torch.min(hue_diff,dim=-1).values
    hue_diff = hue_diff/180
    
    weight_division = est['wrong_weight_d_o'][:,0]/(est['wrong_weight_d_o'][:,0] + est['wrong_weight_il'][:,0] +1e-6)
    loss = masked_loss(data['liv_masks'],weight_division*(hue_diff<1/3))#torch.maximum(torch.ones_like(hue_diff,device=hue_diff.device)-3*hue_diff,torch.zeros_like(hue_diff,device=hue_diff.device)))#*(hue_diff<(1/3))*Mc_loss(hue_diff,k=1.5))
    
    #hinge loss
    
    if not torch.isfinite(loss).all():
        log.warning("hue_loss: non-finite loss detected")
    return loss
BCE_loss = custom_bceloss
class BaseTrainer(abc.ABC):
    def __init__(self,
                 train_data_loader: Iterable,
                 loss_weights:Dict,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs):
        self.loss_weight = loss_weights
        self.train_data_loader = train_data_loader
        self.num_steps = num_steps
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        self.device = device
        self.eval_batch_size = kwargs.get('eval_batch_size', 8129)
        self.extra_args = kwargs
        self.timer = CudaTimer(enabled=False)

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step: Optional[int] = None
        self.loss_info: Optional[Dict[str, EMA]] = None

        self.model = self.init_model(**self.extra_args)
        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.regularizers = self.init_regularizers(**self.extra_args)
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)
        self.change_hue = kwargs['change_hue']
        
        self.model.to(self.device)

    @abc.abstractmethod
    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    @abc.abstractmethod
    def render_wo_il_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa

    @abc.abstractmethod
    def render_fig8_step(self, data, time,**kwargs):
        self.model.eval()
        return None
    
    @abc.abstractmethod

    def train_step(self, data, **kwargs) -> bool:
        self.model.train()
        if self.global_step > self.change_hue:
            self.model.l_field.hues.requires_grad = True
        else:
            self.model.l_field.hues.requires_grad = False

        data = self._move_data_to_device(data)
        if "timestamps" not in data:
            data["timestamps"] = None
        self.timer.check("move-to-device")
        #https://github.com/NVlabs/tiny-cuda-nn/issues/179 ISSUE:!
                    
        
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            self.update_Hues()
            if (self.model.l_field.hues>360).any() or (self.model.l_field.hues<0.0).any():
                log.warning("Hue values outside valid [0, 360] interval")

            
            fwd_out = self.model(
                data['rays_o'], data['rays_d'], bg_color=data['bg_color'],
                near_far=data['near_fars'], timestamps=data['timestamps'])
            self.timer.check("model-forward")
            recon_loss =  self.criterion(fwd_out['liv_rgb'], data['liv_imgs'])
 
            if "warped_rgb" in data:
                rendered_dynamics = self.model.reh_dynamic_render(data['sque_rays_o'],data['sque_rays_d'],data['bg_color'],data['near_fars'],data['sampled_times'])
                d_squential_loss = masked_loss(data['sque_conf']*data['liv_masks'][:,None,:],torch.pow(fwd_out['liv_d_only_rgb'][:,None,:]-rendered_dynamics,2)/rendered_dynamics.shape[-2])
                d_supervision_loss = masked_loss(data['warped_conf']*data['liv_masks'],torch.pow(fwd_out['liv_d_only_rgb']-data['warped_rgb'],2))
                
            ones = torch.ones(data['liv_masks'].shape,dtype=data['liv_masks'].dtype,device=data['liv_masks'].device,requires_grad=False)
            liv_dynamic_loss = masked_loss(data['liv_masks'],torch.pow(fwd_out['liv_rgb']-data['liv_imgs'],2))
            liv_static_loss = masked_loss(ones-data['liv_masks'],torch.pow(fwd_out['liv_s_rgb']-data['liv_imgs'],2))
            reh_static_loss = masked_loss((ones-data['reh_masks']),torch.pow(fwd_out['reh_rgb']-data['reh_imgs'],2))
            liv_wo_il_loss = masked_loss((ones-data['reh_masks'])*(ones-data['liv_masks']),torch.pow(data['reh_imgs']-fwd_out['liv_wo_il'],2)) + \
                masked_loss((ones-data['liv_masks']),torch.pow(fwd_out['reh_rgb']-fwd_out['liv_wo_il'],2))

            if not torch.isfinite(recon_loss).all():
                log.warning("Reconstruction loss contains non-finite values")
                recon_loss = recon_loss.nan_to_num(nan=1e-6)
            wrong_weight_loss =  masked_loss((ones-data['liv_masks']),fwd_out['weight_d']).nan_to_num(nan=1e-6) + masked_loss(data['liv_masks'],fwd_out['weight_s']).nan_to_num(nan=1e-6)# + masked_loss(data['reh_masks'],fwd_out['wrong_weight_s']).nan_to_num(nan=1e-6) #+_loss(fwd_out,data,ones)
            #hue_loss_value =  self.loss_weight['hue_loss']*hue_loss(data,fwd_out)
            loss = recon_loss + self.loss_weight['reh_static_loss']*reh_static_loss.nan_to_num(nan=1e-6) + self.loss_weight['liv_static_loss']*liv_static_loss.nan_to_num(nan=1e-6) + self.loss_weight['liv_dynamic_loss']*liv_dynamic_loss.nan_to_num(nan=1e-6)   #+ ill_switch*reh_liv_diff_loss(fwd_out,data,ones)
            loss = loss + self.loss_weight['liv_wo_il_loss']*liv_wo_il_loss
            loss = loss  + self.loss_weight['wrong_weight_loss']*wrong_weight_loss
            #loss = loss #+  hue_loss_value
            loss = loss + self.loss_weight['dynamic_loss']*(d_squential_loss + d_supervision_loss)
            #Intensity smoothenss loss
            #spatial_smooth = fwd_out['spatial_smooth']
            #temporal_smooth = fwd_out['temporal_smooth']
            scale_loss = fwd_out['scale_loss']
            alpha_loss = fwd_out['alpha_loss']
            loss = loss + self.loss_weight['scale_loss']*scale_loss
            loss = loss + self.loss_weight['alpha_loss']*alpha_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, model_out=fwd_out)
                if reg_loss is not None:
                    if not torch.isfinite(reg_loss).all():
                        log.warning("Regularizer %s produced non-finite loss", r)
                        reg_loss = reg_loss.nan_to_num(nan=1e-6)
                    loss = loss + reg_loss
            self.timer.check("regularizaion-forward")
        # Update weights
        self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()
        self.timer.check("backward")
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()
        self.timer.check("scaler-step")

        # Report on losses
        if self.global_step % self.calc_metrics_every == 0:
            with torch.no_grad():
                #acc = torch.sum( ( (fwd_out['dynamic_class']>0.5) == data['liv_masks'])*(fwd_out['dynamic_class']>0.5)  )/(torch.sum(data['liv_masks']))
                d_squential_loss_val = d_squential_loss.item()
                d_supervision_loss_val = d_supervision_loss.item()

                recon_loss_val = recon_loss.item()
                #hue_loss_val = hue_loss_value.item()
                #liv_dynamic_loss_val = liv_dynamic_loss.item()
                liv_static_loss_val = liv_static_loss.item()
                reh_static_loss_val = reh_static_loss.item()
                wrong_weight_loss_val = wrong_weight_loss.item()
                
                #spatial_smooth_val = spatial_smooth.item()
                #temporal_smooth_val = temporal_smooth.item()
                scale_loss_val = scale_loss.item()
                alpha_loss_val = alpha_loss.item()

                
                #bce_loss_val = bce_loss.item()
                self.loss_info[f"mse"].update(recon_loss_val)
                self.loss_info[f"psnr"].update(-10 * math.log10(recon_loss_val))
                self.loss_info[f"d_squ"].update(d_squential_loss_val)
                self.loss_info[f"d_sup"].update(d_supervision_loss_val)
                #self.loss_info[f"spatial"].update(spatial_smooth_val)
                #self.loss_info[f"temp"].update(temporal_smooth_val)
                self.loss_info[f"scale"].update(scale_loss_val)
                self.loss_info[f"alpha"].update(alpha_loss_val)

                #self.loss_info[f"psnr_ld"].update(-10 * math.log10(liv_dynamic_loss_val))
                self.loss_info[f"psnr_ls"].update(-10 * math.log10(liv_static_loss_val))
                self.loss_info[f"psnr_rs"].update(-10 * math.log10(reh_static_loss_val))
                #self.loss_info[f"hue"].update(hue_loss_val)
                self.loss_info[f"w_w"].update(wrong_weight_loss_val)

                for r in self.regularizers:
                    r.report(self.loss_info)

        return scale <= self.gscaler.get_scale()
    @abc.abstractmethod
    def option1_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    @abc.abstractmethod
    def option2_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    @abc.abstractmethod
    def option3_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    @abc.abstractmethod
    def option4_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    @abc.abstractmethod
    def app_hue(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    @abc.abstractmethod
    def app_no_il_and_attention_test(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    @abc.abstractmethod
    def app_ill(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa

    @abc.abstractmethod
    def opt1_test(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    @abc.abstractmethod
    def opt2_test(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    @abc.abstractmethod
    def opt3_test(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    @abc.abstractmethod
    def opt4_test(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    @abc.abstractmethod
    def app_test_hue(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    @abc.abstractmethod
    def app_test_ill(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    @abc.abstractmethod
    def app_brighter(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def app_test_no_light(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    @abc.abstractmethod
    def supp_test_hue(self, data, hue_value, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    @abc.abstractmethod
    def supp_multiple_hues(self,  hue_value, **kwargs) -> MutableMapping[str, torch.Tensor]:
        pass
    
    @abc.abstractmethod
    def supp_frozen_motion_step(self,  time, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    @abc.abstractmethod
    def supp_frozen_light_step(self,  time, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa
    def supp_change_il_map(self,  data, il_map, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa


    def post_step(self, progress_bar):
        self.model.step_after_iter(self.global_step)
        if self.global_step % self.calc_metrics_every == 0:
            progress_bar.set_postfix_str(
                losses_to_postfix(self.loss_info, lr=self.lr), refresh=False)
            for loss_name, loss_val in self.loss_info.items():
                self.writer.add_scalar(f"train/loss/{loss_name}", loss_val.value, self.global_step)
                if self.timer.enabled:
                    tsum = 0.
                    tstr = "Timings: "
                    for tname, tval in self.timer.timings.items():
                        tstr += f"{tname}={tval:.1f}ms  "
                        tsum += tval
                    tstr += f"tot={tsum:.1f}ms"
                    log.info(tstr)
        progress_bar.update(1)
        if self.save_every > -1 and self.global_step % self.save_every == 0:
            self.save_model()

        if (self.valid_every > -1 and self.global_step % self.valid_every == 0 ) or self.global_step==10000:
            #self.opt1_test()
            #self.opt2_test()
            #self.opt3_test()
            self.validate()
            gc.collect()
            self.opt4_test()
            gc.collect()
            self.app_test_hue()
            gc.collect()
            self.app_test_no_light()
            gc.collect()
            self.app_test_ill()    
            gc.collect()

    def pre_epoch(self):
        self.loss_info = self.init_epoch_info()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        log.info(f"Starting training from step {self.global_step + 1}")
        pb = tqdm(initial=self.global_step, total=self.num_steps)
        try:
            self.pre_epoch()
            batch_iter = iter(self.train_data_loader)
            while self.global_step < self.num_steps:
                self.timer.reset()
                self.model.step_before_iter(self.global_step)
                self.global_step += 1
                self.timer.check("step-before-iter")
                try:
                    data = next(batch_iter)
                    self.timer.check("dloader-next")
                except StopIteration:
                    self.pre_epoch()
                    batch_iter = iter(self.train_data_loader)
                    data = next(batch_iter)
                    log.info("Reset data-iterator")

                try:
                    step_successful = self.train_step(data)
                except StopIteration:
                    self.pre_epoch()
                    batch_iter = iter(self.train_data_loader)
                    log.info("Reset data-iterator")
                    step_successful = True

                if step_successful and self.scheduler is not None:
                    self.scheduler.step()
                for r in self.regularizers:
                    r.step(self.global_step)
                self.post_step(progress_bar=pb)
                self.timer.check("after-step")
        finally:
            pb.close()
            self.writer.close()

    def _move_data_to_device(self, data):
        data["rays_o"] = data["rays_o"].to(self.device)
        data["rays_d"] = data["rays_d"].to(self.device)
        data["liv_imgs"] = data["liv_imgs"].to(self.device)
        data["liv_masks"] = data["liv_masks"].to(self.device)
        data["reh_imgs"] = data["reh_imgs"].to(self.device)
        data["reh_masks"] = data["reh_masks"].to(self.device)
        data["near_fars"] = data["near_fars"].to(self.device)
        if "timestamps" in data:
            data["timestamps"] = data["timestamps"].to(self.device)
        if "warped_rgb" in data:
            data["warped_rgb"] = data["warped_rgb"].to(self.device)
            data["sque_rays_o"] = data["sque_rays_o"].to(self.device)
            data["sque_rays_d"] = data["sque_rays_d"].to(self.device)
            data["warped_conf"] = data["warped_conf"].to(self.device)
            data["sque_conf"] = data["sque_conf"].to(self.device)
            data["sampled_times"] = data["sampled_times"].to(self.device)

        bg_color = data["bg_color"]
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.to(self.device)
        data["bg_color"] = bg_color
        return data

    def _normalize_err(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        err = torch.abs(preds - gt)
        err = err.mean(-1, keepdim=True)  # mean over channels
        # normalize between 0, 1 where 1 corresponds to the 90th percentile
        # err = err.clamp_max(torch.quantile(err, 0.9))
        err = self._normalize_01(err)
        return err.repeat(1, 1, 3)

    @staticmethod
    def _normalize_01(t: torch.Tensor) -> torch.Tensor:
        return (t - t.min()) / t.max()

    def _normalize_depth(self, depth: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        return (
            self._normalize_01(depth)
        ).cpu().reshape(img_h, img_w)[..., None]

    def calc_metrics(self, preds: torch.Tensor, gt: torch.Tensor):
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])

        err = (gt - preds) ** 2
        return {
            "mse": torch.mean(err),
            "psnr": metrics.psnr(preds, gt),
            "ssim": metrics.ssim(preds, gt),
            "ms-ssim": metrics.msssim(preds, gt),
            #"alex_lpips": metrics.rgb_lpips(preds, gt, net_name='alex', device=err.device),
            #"vgg_lpips": metrics.rgb_lpips(preds, gt, net_name='vgg', device=err.device)
        }
    def argmax_class2rgb(self, classes:torch.Tensor):
        r_indxs = torch.where(classes==0)
        g_indxs = torch.where(classes==1)
        b_indxs = torch.where(classes==2)
        rgb_classes = torch.zeros((classes.shape[0],3),dtype=torch.float32)
        rgb_classes[r_indxs[0],0] = 1.0
        rgb_classes[g_indxs[0],1] = 1.0
        rgb_classes[b_indxs[0],2] = 1.0
        return rgb_classes

    def evaluate_metrics(self,
                         data: MutableMapping[str, torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,
                         name: Optional[str] = None,
                         save_outputs: bool = True) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
        preds_rgb = (
            preds["liv_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        out_brdf = (
            preds["brdf"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        out_brdf_s = (
            preds["brdf_s"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        out_brdf_d = (
            preds["brdf_d"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        out_d_rgb = (
            preds["liv_d_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        out_d_only_rgb = (
            preds["liv_d_only_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        preds_reh_rgb = (
            preds["reh_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        preds_lighting_illum = (
            preds["lighting_illum"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        preds_lighting_illum_w_alpha = (
            preds["lighting_illum_alpha"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )

        gt = data['liv_imgs']
        if not torch.isfinite(preds_rgb).all():
            log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                        f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=1e-6)
        out_reh = preds_reh_rgb
        out_img = preds_rgb
        summary = dict()

        out_depth = None
        if "depth" in preds:
            out_depth = preds["depth"].cpu().reshape(img_h, img_w)[..., None]
            

        preds.pop("depth")
        # This is used for proposal-depth keys
        for k in preds.keys():
            if "depth" in k:
                prop_depth = preds[k].cpu().reshape(img_h, img_w)[..., None]
                out_depth = torch.cat((out_depth, prop_depth)) if out_depth is not None else prop_depth
        if "illum" in preds:
            out_illum = preds['illum'].cpu().reshape(img_h, img_w)[..., None]

        if gt is not None:
            gt = gt.reshape(img_h, img_w, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            summary.update(self.calc_metrics(preds_rgb, gt))
            out_img = torch.cat((out_img, gt), dim=0)
            out_img = torch.cat((out_img, self._normalize_err(preds_rgb, gt)), dim=0)
            #out_reh = torch.cat((out_reh), dim=0)
        out_brdf_np:np.ndarray = (out_brdf * 255.0).byte().numpy()
        out_brdf_s_np:np.ndarray = (out_brdf_s * 255.0).byte().numpy()
        out_brdf_d_np:np.ndarray = (out_brdf_d * 255.0).byte().numpy()
        preds_lighting_illum_np:np.ndarray = (preds_lighting_illum * 255.0).byte().numpy()
        preds_lighting_illum_w_alpha_np:np.ndarray = (preds_lighting_illum_w_alpha * 255.0).byte().numpy()

        out_reh_np:np.ndarray = (out_reh * 255.0).byte().numpy()
        out_img_np: np.ndarray = (out_img * 255.0).byte().numpy()
        out_d_img_np: np.ndarray = (out_d_rgb * 255.0).byte().numpy()
        out_d_only_img_np: np.ndarray = (out_d_only_rgb * 255.0).byte().numpy()

        out_depth_np: Optional[np.ndarray] = None
        if out_depth is not None:
            out_depth = self._normalize_01(out_depth)
            out_depth_np = (out_depth * 255.0).repeat(1, 1, 3).byte().numpy()
        out_illum_np: Optional[np.ndarray] = None
        if out_depth is not None:
            out_illum = self._normalize_01(out_illum)
            out_illum_np = (out_illum * 255.0).repeat(1, 1, 3).byte().numpy()

        if save_outputs:
            out_name = f"step{self.global_step}-{img_idx}"
            if name is not None and name != "":
                out_name += "-" + name
            write_png(os.path.join(self.log_dir, out_name + ".png"), out_img_np)

            if out_depth is not None:
                depth_name = out_name + "-depth"
                write_png(os.path.join(self.log_dir, depth_name + ".png"), out_depth_np)
        return summary, out_img_np, out_d_img_np, out_d_only_img_np, out_reh_np,out_depth_np, out_brdf_np, out_brdf_s_np, out_brdf_d_np, out_illum_np, preds_lighting_illum_np, preds_lighting_illum_w_alpha_np
    
    def show_applies(self,
                         data: MutableMapping[str, torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,
                         name: Optional[str] = None,
                         save_outputs: bool = True) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
            
        """            "brdf":brdf_all,
            "brdf_s":brdf_s,
            "brdf_d":brdf_d,

        Returns:
            _type_: _description_
        """
        preds_rgb = (
            preds["liv_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        if preds["brdf"] is not None:
            out_brdf = (
                preds["brdf"]
                .reshape(img_h, img_w, 3)
                .cpu()
                .clamp(0, 1)
            )
        out_brdf_s = (
            preds["brdf_s"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        out_brdf_d = (
            preds["brdf_d"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        out_d_rgb = (
            preds["liv_d_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )

        if not torch.isfinite(preds_rgb).all():
            log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                        f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=1e-6)
        out_img = preds_rgb
            
        # This is used for proposal-depth keys
        out_brdf_np:np.ndarray = (out_brdf * 255.0).byte().numpy()
        out_brdf_s_np:np.ndarray = (out_brdf_s * 255.0).byte().numpy()
        out_brdf_d_np:np.ndarray = (out_brdf_d * 255.0).byte().numpy()

        out_img_np: np.ndarray = (out_img * 255.0).byte().numpy()
        out_d_img_np: np.ndarray = (out_d_rgb * 255.0).byte().numpy()

        return  out_img_np, out_d_img_np, out_brdf_np, out_brdf_s_np, out_brdf_d_np
    def show_only_rgb(self,
                         data: MutableMapping[str, torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,
                         name: Optional[str] = None,
                         save_outputs: bool = True) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
            
        """            "brdf":brdf_all,
            "brdf_s":brdf_s,
            "brdf_d":brdf_d,

        Returns:
            _type_: _description_
        """
        preds_rgb = (
            preds["liv_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        if not torch.isfinite(preds_rgb).all():
            log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                        f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=1e-6)
        out_img = preds_rgb
            

        out_img_np: np.ndarray = (out_img * 255.0).byte().numpy()

        return  out_img_np

    def test_shift_time(self,
                         data: MutableMapping[str, torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,
                         name: Optional[str] = None,
                         save_outputs: bool = True) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
        preds_rgb = (
            preds["liv_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        preds_s_rgb = (
            preds["liv_s_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        preds_d_rgb = (
            preds["liv_d_rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        if not torch.isfinite(preds_rgb).all():
            log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                        f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=1e-6)
            
        # This is used for proposal-depth keys

        out_img_np: np.ndarray = (preds_rgb * 255.0).byte().numpy()
        out_s_img_np: np.ndarray = (preds_s_rgb * 255.0).byte().numpy()
        out_d_img_np: np.ndarray = (preds_d_rgb * 255.0).byte().numpy()

        return  out_img_np, out_s_img_np, out_d_img_np
    
    @abc.abstractmethod
    def validate(self):
        pass


    def report_test_metrics(self, scene_metrics: Dict[str, Sequence[float]], extra_name: Optional[str]):
        log_text = f"step {self.global_step}/{self.num_steps}"
        if extra_name is not None:
            log_text += f" | {extra_name}"
        scene_metrics_agg: Dict[str, float] = {}
        for k in scene_metrics:
            ak = f"{k}_{extra_name}"
            scene_metrics_agg[ak] = np.mean(np.asarray(scene_metrics[k])).item()
            log_text += f" | {k}: {scene_metrics_agg[ak]:.4f}"
            self.writer.add_scalar(f"test/{ak}", scene_metrics_agg[ak], self.global_step)

        log.info(log_text)
        return scene_metrics_agg

    def get_save_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step
        }

    def save_model(self):
        model_fname = os.path.join(self.log_dir, str(self.global_step).zfill(5) + f'model.pth')
        log.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.get_save_dict(), model_fname)

    def load_model(self, checkpoint_data, training_needed: bool = True):
        self.model.load_state_dict(checkpoint_data["model"], strict=False)
        log.info("=> Loaded model state from checkpoint")

        if training_needed:
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            log.info("=> Loaded optimizer state from checkpoint")

        if training_needed and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
            log.info("=> Loaded scheduler state from checkpoint")

        self.global_step = checkpoint_data["global_step"]
        log.info(f"=> Loaded step {self.global_step} from checkpoints")

    @abc.abstractmethod
    def init_epoch_info(self) -> Dict[str, EMA]:
        pass

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = self.num_steps
        scheduler_type = kwargs['scheduler_type']
        log.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 f"{max_steps} maximum steps.")
        if scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
        elif scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=512, num_training_steps=max_steps)
        elif scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33)
        elif scheduler_type == "warmup_step":
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer, milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
                num_warmup_steps=512)
        return lr_sched

    def init_optim(self, **kwargs) -> torch.optim.Optimizer:
        optim_type = kwargs['optim_type']
        if optim_type == 'adam':
            optim = torch.optim.Adam(params=self.model.get_params(kwargs['lr']), eps=1e-15, weight_decay=1e-15)
        else:
            raise NotImplementedError()
            
        return optim
    def update_Hues(self):
        self.model.l_field.hues.data[self.model.l_field.hues.data<0.0] = self.model.l_field.hues.data[self.model.l_field.hues.data<0.0] + 360

        self.model.l_field.hues.data = self.model.l_field.hues.data  % 360
    @abc.abstractmethod
    def init_model(self, **kwargs) -> torch.nn.Module:
        pass

    def get_regularizers(self, **kwargs) -> Sequence[Regularizer]:
        return ()

    def init_regularizers(self, **kwargs):
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in self.get_regularizers(**kwargs) if r.weight > 0]
        return regularizers

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def calc_metrics_every(self):
        return 1


def losses_to_postfix(loss_dict: Dict[str, EMA], lr: Optional[float]) -> str:
    pfix = [f"{lname}={lval}" for lname, lval in loss_dict.items()]
    if lr is not None:
        pfix.append(f"lr={lr:.2e}")
    return "  ".join(pfix)


def init_dloader_random(_):
    seed = torch.initial_seed() % 2**32  # worker-specific seed initialized by pytorch
    np.random.seed(seed)
    random.seed(seed)


def initialize_model(
        runner: ['KpopTrainer'],
        **kwargs) -> LowrankModel:
    """Initialize a `LowrankModel` according to the **kwargs parameters.

    Args:
        runner: The runner object which will hold the model.
                Needed here to fetch dataset parameters.
        **kwargs: Extra parameters to pass to the model

    Returns:
        Initialized LowrankModel.
    """
    extra_args = copy(kwargs)
    extra_args.pop('global_scale', None)
    extra_args.pop('global_translation', None)
    
    dset = runner.test_dataset
    try:
        global_translation = dset.global_translation
    except AttributeError:
        global_translation = None
    try:
        global_scale = dset.global_scale
    except AttributeError:
        global_scale = None

    hues = torch.load(os.path.join(extra_args['data_dirs'][0], "temporal_hues.pt"))
    model = LowrankModel(
        hues = hues,
        aabb=dset.scene_bbox,
        is_ndc=dset.is_ndc,
        is_contracted=dset.is_contracted,
        global_scale=global_scale,
        global_translation=global_translation,
        **extra_args)

    log.info(f"Initialized {model.__class__} model with "
             f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, "
             f"using ndc {model.is_ndc} and contraction {model.is_contracted}. "
             f"Linear decoder: {model.D_linear_decoder}.")
    return model
