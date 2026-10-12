from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from models.density_fields import KPlaneDensityField
from models.dynamic_field import D_field
from models.static_field import S_field
from models.lplane_field import Lplane_field
from models.attention import Attention_base, MLP_base, MLP_base_w_brdf, MLP_base_w_full
from models.tone_mapping import Main_tonemapping_RGB, Main_tonemapping_Mono, NeRV_tonemapping
from ops.activations import init_density_activation
from raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)
from raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from utils.timer import CudaTimer

class LowrankModel(nn.Module):
    def __init__(self,
                 hues: torch.Tensor,
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,#?
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 use_motion:bool = False,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 # proposal-sampling arguments
                 num_proposal_iterations: int = 1,
                 use_same_proposal_network: bool = False,
                 proposal_net_args_list: List[Dict] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 num_samples: Optional[int] = None,
                 single_jitter: bool = False,
                 proposal_warmup: int = 5000,
                 proposal_update_every: int = 5,
                 use_proposal_weight_anneal: bool = True,
                 proposal_weights_anneal_max_num_iters: int = 1000,
                 proposal_weights_anneal_slope: float = 10.0,
                 num_frames: int = None,
                 # lightning field
                 L_concat_features_across_scales: bool = False,
                 L_linear_decoder_layers: Optional[int] = 1,
                 L_grid_config: Union[str, List[Dict]] = None,
                 L_basis_color_dim : int = 32,
                 L_apperance_embdding_dim: int = 32,
                 L_decoder_layer: int = 5,
                 k: int = 5,
                 # static network
                 S_density_activation: Optional[str] = 'trunc_exp',
                 S_concat_features_across_scales: bool = False,
                 S_linear_decoder: bool = True,
                 S_linear_decoder_layers: Optional[int] = 1,
                 S_grid_config: Union[str, List[Dict]] = None,
                 S_basis_color_dim : int = 16,
                 S_decoder_layer: int = 5,
                 # dynamic network
                 D_density_activation: Optional[str] = 'trunc_exp',
                 D_concat_features_across_scales: bool = False,
                 D_linear_decoder: bool = True,
                 D_linear_decoder_layers: Optional[int] = 1,
                 D_grid_config: Union[str, List[Dict]] = None,
                 D_basis_color_dim : int = 4,
                 D_decoder_layer: int = 5,
                 # Tone mapping function 
                 tone_mapping_hidden_dim: int = 16,
                 tone_mapping_option: str='rgb',#rgb or mono                 
                 # intensity_decoder
                 intensity_decoder_type: str='attention',#attention or mlp
                 intensity_decoder_hidden_dim: int = 16,
                 intensity_decoder_out_dim: int = 3, #3 or 1 ( rgb or mono)
                 **kwargs,
                 ):
        super().__init__()
        
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.use_motion = use_motion
        self.hues =hues
        self.is_contracted = is_contracted
        #for lightning network
        self.L_concat_features_across_scales = L_concat_features_across_scales
        self.L_linear_decoder_layers = L_linear_decoder_layers
        self.L_grid_config = L_grid_config
        self.L_basis_color_dim = L_basis_color_dim
        self.L_apperance_embdding_dim = L_apperance_embdding_dim
        self.L_decoder_layer = L_decoder_layer
        #for static network
        self.S_concat_features_across_scales = S_concat_features_across_scales
        self.S_linear_decoder = S_linear_decoder
        self.S_linear_decoder_layers = S_linear_decoder_layers
        self.S_density_act = init_density_activation(S_density_activation)
        self.S_grid_config = S_grid_config
        self.S_basis_color_dim = S_basis_color_dim
        self.S_decoder_layer = S_decoder_layer
        #for dynamic network
        self.D_concat_features_across_scales = D_concat_features_across_scales
        self.D_linear_decoder = D_linear_decoder
        self.D_linear_decoder_layers = D_linear_decoder_layers
        self.D_density_act = init_density_activation(D_density_activation)
        self.D_grid_config = D_grid_config
        self.D_basis_color_dim = D_basis_color_dim
        self.D_decoder_layer = D_decoder_layer

        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)
        self.d_field = D_field(
            aabb,
            grid_config=self.D_grid_config,
            concat_features_across_scales=self.D_concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.D_density_act,
            linear_decoder=self.D_linear_decoder,
            linear_decoder_layers=self.D_linear_decoder_layers,
            num_frames = num_frames,
        )
        self.s_field = S_field(
            aabb,
            grid_config=self.S_grid_config,
            concat_features_across_scales=self.S_concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.S_density_act,
            linear_decoder_layers=self.S_linear_decoder_layers,
            linear_decoder=self.S_linear_decoder,
        )
        self.l_field = Lplane_field(
            aabb,
            grid_config=self.L_grid_config,
            concat_features_across_scales=self.D_concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            spatial_distortion=self.spatial_distortion,
            ratio_activation=torch.sigmoid,
            linear_decoder=self.D_linear_decoder,
            linear_decoder_layers=self.D_linear_decoder_layers,
            k = k,
            hues = self.hues,
            num_frames = num_frames,
        )
        self.tone_mapping_option = tone_mapping_option
        if tone_mapping_option=='rgb':
            self.tone_mapping = Main_tonemapping_RGB(tone_mapping_hidden_dim)
        elif tone_mapping_option=='mono':
            self.tone_mapping = Main_tonemapping_Mono(tone_mapping_hidden_dim)
        elif tone_mapping_option=='nerv':
            self.tone_mapping = NeRV_tonemapping()

        else:
            raise ValueError(f"Unknown tone_mapping_option: {tone_mapping_option}")
        
        if intensity_decoder_type=='attention':
            self.interact_decoder = Attention_base(intensity_decoder_hidden_dim,intensity_decoder_out_dim)
        elif intensity_decoder_type=='mlp':
            self.interact_decoder = MLP_base( intensity_decoder_hidden_dim,intensity_decoder_out_dim)
        elif intensity_decoder_type=='mlp_full':
            self.interact_decoder = MLP_base_w_full( intensity_decoder_hidden_dim,intensity_decoder_out_dim)

        
        # Initialize proposal-sampling nets
        self.density_fns = []
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = torch.nn.ModuleList()
        if use_same_proposal_network:
            assert len(self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.D_density_act, linear_decoder=self.D_linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)])
        else:#use different proposal network...
            for i in range(self.num_proposal_iterations):
                prop_net_args = self.proposal_net_args_list[min(i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.D_density_act, linear_decoder=self.D_linear_decoder, **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for network in self.proposal_networks])

        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.proposal_warmup], [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )

    def step_before_iter(self, step):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)

        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def reh_dynamic_render(self, rays_o, rays_d, bg_color, near_far, timestamps):
        """
        rays_o : [batch, interval, 3]
        rays_d : [batch, interval, 3]
        timestamps : [batch, interval]
        near_far : [batch, 2]
        """
        interval = rays_o.shape[1]
        rays_o = rays_o.view(-1,rays_o.shape[-1])
        rays_d = rays_d.view(-1,rays_d.shape[-1])
        timestamps = timestamps.view(-1) 
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears[0]
            fars = ones * fars[0]

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples( ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        _, d_weights, _, _, _ = ray_samples.get_weights_decomp(static_density,dynamic_density)
        d_rgb = self.render_rgb(rgb=self.tone_mapping(dynamic_radiance), weights=d_weights, bg_color=bg_color)
        # d_rgbs should be : [batch, interval, 3]
        d_rgb = d_rgb.view(-1,interval,d_rgb.shape[-1])
        return d_rgb
    
    
    def render_change_ill(self, rays_o, rays_d, bg_color, hue_value, il_ratio, near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        light_field_out = self.l_field.forward_change_hue(hue_value,il_ratio,ray_samples.get_positions(), ray_bundle.directions, timestamps)
        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color = light_field_out['l_color']
        

        s_weights, d_weights, s_only_weights, d_only_weights, total_weights = ray_samples.get_weights_decomp(static_density,dynamic_density)

        ones = torch.ones(static_radiance.shape,dtype = static_radiance.dtype,device=static_radiance.device)# 4096 1
        static_brdf = self.interact_decoder(static_radiance,l_color)
        dynamic_brdf = self.interact_decoder(dynamic_radiance,l_color)
        static_rgb_liv = self.tone_mapping(static_radiance+static_brdf*l_color)
        dynamic_rgb_liv = self.tone_mapping(dynamic_radiance+dynamic_brdf*l_color)



        liv_s_rgb = self.render_rgb(rgb=static_rgb_liv, weights=s_weights, bg_color=None)
        liv_d_rgb = self.render_rgb(rgb=dynamic_rgb_liv, weights=d_weights, bg_color=None)
        liv_rgb = liv_s_rgb + liv_d_rgb
        brdf_s = self.render_rgb(rgb=static_brdf, weights=s_weights, bg_color=None)
        brdf_d = self.render_rgb(rgb=dynamic_brdf, weights=d_weights, bg_color=None)
        brdf_all = brdf_s + brdf_d
        outputs = {
            "liv_rgb": liv_rgb,
            "liv_s_rgb": liv_s_rgb,
            "liv_d_rgb": liv_d_rgb,
            "brdf": brdf_all,
            "brdf_s": brdf_s,
            "brdf_d": brdf_d,
        }
        return outputs

    def supp_il_map(self, rays_o, rays_d, bg_color, il_map, near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        light_field_out = self.l_field.forward_il_map(il_map,ray_samples.get_positions(), ray_bundle.directions, timestamps)
        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color = light_field_out['l_color']
        

        s_weights, d_weights, s_only_weights, d_only_weights, total_weights = ray_samples.get_weights_decomp(static_density,dynamic_density)

        ones = torch.ones(static_radiance.shape,dtype = static_radiance.dtype,device=static_radiance.device)# 4096 1
        static_brdf = self.interact_decoder(static_radiance,l_color)
        dynamic_brdf = self.interact_decoder(dynamic_radiance,l_color)
        static_rgb_liv = self.tone_mapping(static_radiance+static_brdf*l_color)
        dynamic_rgb_liv = self.tone_mapping(dynamic_radiance+dynamic_brdf*l_color)



        liv_s_rgb = self.render_rgb(rgb=static_rgb_liv, weights=s_weights, bg_color=None)
        liv_d_rgb = self.render_rgb(rgb=dynamic_rgb_liv, weights=d_weights, bg_color=None)
        liv_rgb = liv_s_rgb + liv_d_rgb
        brdf_s = self.render_rgb(rgb=static_brdf, weights=s_weights, bg_color=None)
        brdf_d = self.render_rgb(rgb=dynamic_brdf, weights=d_weights, bg_color=None)
        brdf_all = brdf_s + brdf_d
        outputs = {
            "liv_rgb": liv_rgb,
            "liv_s_rgb": liv_s_rgb,
            "liv_d_rgb": liv_d_rgb,
            "brdf": brdf_all,
            "brdf_s": brdf_s,
            "brdf_d": brdf_d,
        }
        return outputs

    def test_shift_time_opt1(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, obj_timestamps, light_timestamps):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, obj_timestamps)
        light_field_out_opt1 = self.l_field.shift_time_opt1(ray_samples.get_positions(), ray_bundle.directions, light_timestamps, obj_timestamps)

        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color_opt1 = light_field_out_opt1['l_color']


        s_weights, d_weights, s_only_weights, d_only_weights, total_weights = ray_samples.get_weights_decomp(static_density,dynamic_density)


        ones = torch.ones(static_radiance.shape,dtype = static_radiance.dtype,device=static_radiance.device)# 4096 1
        
        static_rgb_liv_opt1 = self.tone_mapping(static_radiance+self.interact_decoder(static_radiance,l_color_opt1)*l_color_opt1)
        dynamic_rgb_liv_opt1 = self.tone_mapping(dynamic_radiance+self.interact_decoder(dynamic_radiance,l_color_opt1)*l_color_opt1)


        ray_samples_list.append(ray_samples)
        liv_s_rgb_op1 = self.render_rgb(rgb=static_rgb_liv_opt1, weights=s_weights, bg_color=None)
        liv_d_rgb_opt1 = self.render_rgb(rgb=dynamic_rgb_liv_opt1, weights=d_weights, bg_color=None)
        liv_rgb_opt1 = liv_s_rgb_op1 + liv_d_rgb_opt1

        return {
            "liv_rgb": liv_rgb_opt1,
            "liv_s_rgb": liv_s_rgb_op1,
            "liv_d_rgb": liv_d_rgb_opt1,
        }

    def test_shift_time_opt2(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, obj_timestamps, light_timestamps):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, obj_timestamps)
        light_field_out_opt = self.l_field.shift_time_opt2(ray_samples.get_positions(), ray_bundle.directions, light_timestamps, obj_timestamps)

        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color_opt = light_field_out_opt['l_color']


        s_weights, d_weights, s_only_weights, d_only_weights, total_weights = ray_samples.get_weights_decomp(static_density,dynamic_density)


        ones = torch.ones(static_radiance.shape,dtype = static_radiance.dtype,device=static_radiance.device)# 4096 1
        
        static_rgb_liv_opt = self.tone_mapping(static_radiance+self.interact_decoder(static_radiance,l_color_opt)*l_color_opt)
        dynamic_rgb_liv_opt = self.tone_mapping(dynamic_radiance+self.interact_decoder(dynamic_radiance,l_color_opt)*l_color_opt)


        ray_samples_list.append(ray_samples)
        liv_s_rgb_opt = self.render_rgb(rgb=static_rgb_liv_opt, weights=s_weights, bg_color=None)
        liv_d_rgb_opt = self.render_rgb(rgb=dynamic_rgb_liv_opt, weights=d_weights, bg_color=None)
        liv_rgb_opt = liv_s_rgb_opt + liv_d_rgb_opt

        # These use a lot of GPU memory, so we avoid storing them for eval.
        return {
            "liv_rgb": liv_rgb_opt,
            "liv_s_rgb": liv_s_rgb_opt,
            "liv_d_rgb": liv_d_rgb_opt,
        }

    def test_shift_time_opt3(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, obj_timestamps, light_timestamps):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, obj_timestamps)
        light_field_out_opt = self.l_field.shift_time_opt3(ray_samples.get_positions(), ray_bundle.directions, light_timestamps, obj_timestamps)

        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color_opt = light_field_out_opt['l_color']


        s_weights, d_weights, s_only_weights, d_only_weights, total_weights = ray_samples.get_weights_decomp(static_density,dynamic_density)


        ones = torch.ones(static_radiance.shape,dtype = static_radiance.dtype,device=static_radiance.device)# 4096 1
        
        static_rgb_liv_opt = self.tone_mapping(static_radiance+self.interact_decoder(static_radiance,l_color_opt)*l_color_opt)
        dynamic_rgb_liv_opt = self.tone_mapping(dynamic_radiance+self.interact_decoder(dynamic_radiance,l_color_opt)*l_color_opt)


        ray_samples_list.append(ray_samples)
        liv_s_rgb_opt = self.render_rgb(rgb=static_rgb_liv_opt, weights=s_weights, bg_color=None)
        liv_d_rgb_opt = self.render_rgb(rgb=dynamic_rgb_liv_opt, weights=d_weights, bg_color=None)
        liv_rgb_opt = liv_s_rgb_opt + liv_d_rgb_opt

        # These use a lot of GPU memory, so we avoid storing them for eval.
        return {
            "liv_rgb": liv_rgb_opt,
            "liv_s_rgb": liv_s_rgb_opt,
            "liv_d_rgb": liv_d_rgb_opt,
        }

    def test_shift_time_opt4(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, obj_timestamps, light_timestamps):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, obj_timestamps)
        light_field_out_opt = self.l_field(ray_samples.get_positions(), ray_bundle.directions, light_timestamps)

        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color_opt = light_field_out_opt['l_color']


        s_weights, d_weights, s_only_weights, d_only_weights, total_weights = ray_samples.get_weights_decomp(static_density,dynamic_density)


        ones = torch.ones(static_radiance.shape,dtype = static_radiance.dtype,device=static_radiance.device)# 4096 1
        
        static_rgb_liv_opt = self.tone_mapping(static_radiance+self.interact_decoder(static_radiance,l_color_opt)*l_color_opt)
        dynamic_rgb_liv_opt = self.tone_mapping(dynamic_radiance+self.interact_decoder(dynamic_radiance,l_color_opt)*l_color_opt)


        ray_samples_list.append(ray_samples)
        liv_s_rgb_opt = self.render_rgb(rgb=static_rgb_liv_opt, weights=s_weights, bg_color=None)
        liv_d_rgb_opt = self.render_rgb(rgb=dynamic_rgb_liv_opt, weights=d_weights, bg_color=None)
        liv_rgb_opt = liv_s_rgb_opt + liv_d_rgb_opt

        # These use a lot of GPU memory, so we avoid storing them for eval.
        return {
            "liv_rgb": liv_rgb_opt,
            "liv_s_rgb": liv_s_rgb_opt,
            "liv_d_rgb": liv_d_rgb_opt,
        }

    def test_attention(self,rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        light_field_out = self.l_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color, il_hues = light_field_out['l_color'], light_field_out['hues'] 
        one_out, zero_out = self.interact_decoder.test_similarity(static_radiance,l_color)
        one_out2, zero_out2 = self.interact_decoder.test_similarity(dynamic_radiance,l_color)
        one_out = torch.concat([one_out,one_out2],dim=0)
        zero_out = torch.concat([zero_out,zero_out2],dim=0)

        return {"one_out":one_out,
                "zero_out": zero_out
        }
    def render_wo_il(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        light_field_out = self.l_field.forward_wo_il(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color, il_hues = light_field_out['l_color'], light_field_out['hues'] 

        

        s_weights, d_weights, s_only_weights, d_only_weights, total_weights = ray_samples.get_weights_decomp(static_density,dynamic_density)
        weights_list.append(s_weights )
        weights_list.append(d_weights )
        weights_list.append(total_weights)

        intensity = light_field_out['intensity']
        if self.training:
            scale_loss = torch.mean(torch.abs(torch.mean(intensity)-torch.mean(static_radiance.clone().detach())) + torch.abs(torch.mean(intensity)-torch.mean(dynamic_radiance.clone().detach())))
        ones = torch.ones(static_radiance.shape,dtype = static_radiance.dtype,device=static_radiance.device)# 4096 1
        
        static_brdf = self.interact_decoder(static_radiance.clone().detach(),l_color.clone().detach())
        dynamic_brdf = self.interact_decoder(dynamic_radiance.clone().detach(),l_color.clone().detach())
        static_rgb_liv = self.tone_mapping(static_radiance+static_brdf*l_color)
        dynamic_rgb_liv = self.tone_mapping(dynamic_radiance+dynamic_brdf*l_color)
        alpha_loss = torch.mean(-torch.log(static_brdf+1e-6) -torch.log(dynamic_brdf+1e-6))



        ray_samples_list.append(ray_samples)
        liv_s_rgb = self.render_rgb(rgb=static_rgb_liv, weights=s_weights, bg_color=None)
        liv_d_rgb = self.render_rgb(rgb=dynamic_rgb_liv, weights=d_weights, bg_color=None)

        tonemapped_dynamic_radiance = self.tone_mapping(dynamic_radiance)
        tonemapped_static_radiance = self.tone_mapping(static_radiance)

        liv_d_only_rgb = self.render_rgb(rgb=tonemapped_dynamic_radiance, weights=d_weights, bg_color=None) 


        wrong_weight_s = torch.sum(s_weights ,dim=-2) # 4096* 48 *1
        wrong_weight_d = torch.sum(d_weights,dim=-2)
        
        liv_rgb = liv_s_rgb + liv_d_rgb
        
        
        
        reh_rgb = self.render_rgb(rgb=tonemapped_static_radiance , weights=s_only_weights, bg_color=None)
        static_wo_ill = self.render_rgb(rgb=tonemapped_static_radiance, weights=s_weights, bg_color=None)
        dynamic_wo_ill = self.render_rgb(rgb=tonemapped_dynamic_radiance, weights=d_weights, bg_color=None)
        liv_wo_il_rgb = static_wo_ill + dynamic_wo_ill
        if not self.training:
            dynamic_light = self.tone_mapping(l_color)
            static_light = self.tone_mapping(l_color)

            illumination = self.render_rgb(intensity, total_weights, bg_color=None)
            lighting_illumination_s = self.render_rgb(static_light, weights=s_weights, bg_color=None)
            lighting_illumination_d = self.render_rgb(dynamic_light, weights=d_weights, bg_color=None)
            lighting_illumination =lighting_illumination_s +  lighting_illumination_d
            light_illum_w_alpha_s = self.tone_mapping(static_brdf*l_color)
            light_illum_w_alpha_d = self.tone_mapping(dynamic_brdf*l_color)

            lighting_illum_w_brdf_s = self.render_rgb(light_illum_w_alpha_s, weights=s_weights, bg_color=None)
            lighting_illum_w_brdf_d = self.render_rgb(light_illum_w_alpha_d, weights=d_weights, bg_color=None)
            lighting_illum_w_brdf =lighting_illum_w_brdf_s  + lighting_illum_w_brdf_d
        depth = self.render_depth(weights=total_weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        brdf_s = self.render_rgb(rgb=static_brdf, weights=s_weights, bg_color=None)
        brdf_d = self.render_rgb(rgb=dynamic_brdf, weights=d_weights, bg_color=None)
        brdf_all = brdf_s + brdf_d
        if self.training:
            outputs = {
                "reh_rgb": reh_rgb,
                'weight_s': wrong_weight_s,
                'weight_d': wrong_weight_d,
                'hues': il_hues,
                'liv_wo_il': liv_wo_il_rgb,
                "liv_rgb": liv_rgb,
                "liv_s_rgb": liv_s_rgb,
                "liv_d_rgb": liv_d_rgb,
                "liv_d_only_rgb": liv_d_only_rgb,
                "depth": depth,
                "brdf": brdf_all,
                "brdf_s":brdf_s,
                "brdf_d":brdf_d,
            }
        if not self.training:
            outputs = {
                "reh_rgb": reh_rgb,
                'weight_s': wrong_weight_s,
                'weight_d': wrong_weight_d,
                'hues': il_hues,
                'liv_wo_il': liv_wo_il_rgb,
                "liv_rgb": liv_rgb,
                "liv_s_rgb": liv_s_rgb,
                "liv_d_rgb": liv_d_rgb,
                "liv_d_only_rgb": liv_d_only_rgb,
                "depth": depth,
                "brdf": brdf_all,
                "brdf_s":brdf_s,
                "brdf_d":brdf_d,
                'lighting_illum':lighting_illumination,
                'lighting_illum_alpha': lighting_illum_w_brdf,
                'illum': illumination,
            }


        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            outputs["scale_loss"] = scale_loss
            outputs["alpha_loss"] = alpha_loss

        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    
    def forward(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=self.density_fns)
        static_field_out = self.s_field(ray_samples.get_positions(), ray_bundle.directions)
        dynamic_field_out = self.d_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        light_field_out = self.l_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        static_radiance, static_density = static_field_out["rgb"], static_field_out["density"]
        dynamic_radiance, dynamic_density = dynamic_field_out["rgb"], dynamic_field_out["density"]
        l_color, il_hues = light_field_out['l_color'], light_field_out['hues'] 

        

        s_weights, d_weights, s_only_weights, d_only_weights, total_weights = ray_samples.get_weights_decomp(static_density,dynamic_density)
        weights_list.append(s_weights )
        weights_list.append(d_weights )
        weights_list.append(total_weights)

        intensity = light_field_out['intensity']
        if self.training:
            scale_loss = torch.mean(torch.abs(torch.mean(intensity)-torch.mean(static_radiance.clone().detach())) + torch.abs(torch.mean(intensity)-torch.mean(dynamic_radiance.clone().detach())))
        ones = torch.ones(static_radiance.shape,dtype = static_radiance.dtype,device=static_radiance.device)# 4096 1
        
        static_brdf = self.interact_decoder(static_radiance.clone().detach(),l_color.clone().detach())
        dynamic_brdf = self.interact_decoder(dynamic_radiance.clone().detach(),l_color.clone().detach())
        static_rgb_liv = self.tone_mapping(static_radiance+static_brdf*l_color)
        dynamic_rgb_liv = self.tone_mapping(dynamic_radiance+dynamic_brdf*l_color)
        alpha_loss = torch.mean(-torch.log(static_brdf+1e-6) -torch.log(dynamic_brdf+1e-6))



        ray_samples_list.append(ray_samples)
        liv_s_rgb = self.render_rgb(rgb=static_rgb_liv, weights=s_weights, bg_color=None)
        liv_d_rgb = self.render_rgb(rgb=dynamic_rgb_liv, weights=d_weights, bg_color=None)

        tonemapped_dynamic_radiance = self.tone_mapping(dynamic_radiance)
        tonemapped_static_radiance = self.tone_mapping(static_radiance)

        liv_d_only_rgb = self.render_rgb(rgb=tonemapped_dynamic_radiance, weights=d_weights, bg_color=None) 


        wrong_weight_s = torch.sum(s_weights ,dim=-2) # 4096* 48 *1
        wrong_weight_d = torch.sum(d_weights,dim=-2)
        
        liv_rgb = liv_s_rgb + liv_d_rgb
        
        
        
        reh_rgb = self.render_rgb(rgb=tonemapped_static_radiance , weights=s_only_weights, bg_color=None)
        static_wo_ill = self.render_rgb(rgb=tonemapped_static_radiance, weights=s_weights, bg_color=None)
        dynamic_wo_ill = self.render_rgb(rgb=tonemapped_dynamic_radiance, weights=d_weights, bg_color=None)
        liv_wo_il_rgb = static_wo_ill + dynamic_wo_ill
        if not self.training:
            dynamic_light = self.tone_mapping(l_color)
            static_light = self.tone_mapping(l_color)

            illumination = self.render_rgb(intensity, total_weights, bg_color=None)
            lighting_illumination_s = self.render_rgb(static_light, weights=s_weights, bg_color=None)
            lighting_illumination_d = self.render_rgb(dynamic_light, weights=d_weights, bg_color=None)
            lighting_illumination =lighting_illumination_s +  lighting_illumination_d
            light_illum_w_alpha_s = self.tone_mapping(static_brdf*l_color)
            light_illum_w_alpha_d = self.tone_mapping(dynamic_brdf*l_color)

            lighting_illum_w_brdf_s = self.render_rgb(light_illum_w_alpha_s, weights=s_weights, bg_color=None)
            lighting_illum_w_brdf_d = self.render_rgb(light_illum_w_alpha_d, weights=d_weights, bg_color=None)
            lighting_illum_w_brdf =lighting_illum_w_brdf_s  + lighting_illum_w_brdf_d
        depth = self.render_depth(weights=total_weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        brdf_s = self.render_rgb(rgb=static_brdf, weights=s_weights, bg_color=None)
        brdf_d = self.render_rgb(rgb=dynamic_brdf, weights=d_weights, bg_color=None)
        brdf_all = brdf_s + brdf_d
        if self.training:
            outputs = {
                "reh_rgb": reh_rgb,
                'weight_s': wrong_weight_s,
                'weight_d': wrong_weight_d,
                'hues': il_hues,
                'liv_wo_il': liv_wo_il_rgb,
                "liv_rgb": liv_rgb,
                "liv_s_rgb": liv_s_rgb,
                "liv_d_rgb": liv_d_rgb,
                "liv_d_only_rgb": liv_d_only_rgb,
                "depth": depth,
                "brdf": brdf_all,
                "brdf_s":brdf_s,
                "brdf_d":brdf_d,
            }
        if not self.training:
            outputs = {
                "reh_rgb": reh_rgb,
                'weight_s': wrong_weight_s,
                'weight_d': wrong_weight_d,
                'hues': il_hues,
                'liv_wo_il': liv_wo_il_rgb,
                "liv_rgb": liv_rgb,
                "liv_s_rgb": liv_s_rgb,
                "liv_d_rgb": liv_d_rgb,
                "liv_d_only_rgb": liv_d_only_rgb,
                "depth": depth,
                "brdf": brdf_all,
                "brdf_s":brdf_s,
                "brdf_d":brdf_d,
                'lighting_illum':lighting_illumination,
                'lighting_illum_alpha': lighting_illum_w_brdf,
                'illum': illumination,
            }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            outputs["scale_loss"] = scale_loss
            outputs["alpha_loss"] = alpha_loss

        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    
    def get_params(self, lr: float):
        s_params = self.s_field.get_params()
        d_params = self.d_field.get_params()
        l_params = self.l_field.get_params()
        decoder_params = self.interact_decoder.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = s_params["field"] + d_params["field"]  + l_params["field"] + [p for pnp in pn_params for p in pnp["field"]]

        if self.tone_mapping_option!='nerv':
            tone_mapping_params = self.tone_mapping.get_params()
            nn_params = s_params["nn"] + d_params["nn"] + l_params["nn"]  + decoder_params["nn"] + tone_mapping_params["nn"]  + [p for pnp in pn_params for p in pnp["nn"]] 
            other_params = s_params["other"] + d_params["other"] + l_params["other"]  + decoder_params["other"]  + tone_mapping_params["other"]  + [p for pnp in pn_params for p in pnp["other"]] # good. max 3
        else:
            nn_params = s_params["nn"] + d_params["nn"] + l_params["nn"]  + decoder_params["nn"]   + [p for pnp in pn_params for p in pnp["nn"]] 
            other_params = s_params["other"] + d_params["other"] + l_params["other"]  + decoder_params["other"]  + [p for pnp in pn_params for p in pnp["other"]] # good. max 3

        hues = l_params["hues"]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
            {"params": hues, "lr": lr}
        ]
