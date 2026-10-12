import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn

from ops.interpolation import grid_sample_wrapper
from raymarching.spatial_distortions import SpatialDistortion


def HSV2RGB(H:torch.Tensor,S:torch.Tensor,V:torch.Tensor):
    # batchsize, 1
    #https://www.had2know.org/technology/hsv-rgb-conversion-formula-calculator.html

    M = 255*V 
    m = M*(1-S)
    z = (M-m) * (1-torch.abs((H/60)%2 -1 ))
    #6 cases
    rgb = torch.where(H < 60, torch.cat([M, z, torch.zeros_like(M)], dim=1),
                      torch.where(H < 120, torch.cat([z, M, torch.zeros_like(M)], dim=1),
                                  torch.where(H < 180, torch.cat([torch.zeros_like(M), M, z], dim=1),
                                              torch.where( H < 240, torch.cat([torch.zeros_like(M), z, M], dim=1),
                                                          torch.where(H < 300, torch.cat([z, torch.zeros_like(M), M], dim=1),
                                                                      torch.cat([M, torch.zeros_like(M), z], dim=1))))))

    rgb = rgb + m
    rgb = rgb/255
    return rgb

def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))  # Generate all axis-pair combinations, e.g. 4D (X,Y,Z,T) => ((X,Y), ...)
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        )) # => [1, out_dim, resolution values for axes in coo_comb (reversed order)]
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    try:
        for scale_id, grid in enumerate(ms_grids[:num_levels]):
            interp_space = 1.
            for ci, coo_comb in enumerate(coo_combs):
                # interpolate in plane
                feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                try:

                    interp_out_plane = (
                        grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                        .view(-1, feature_dim)
                    )
                    # compute product over planes
                    interp_space = interp_space * interp_out_plane
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception:
                    pass

            # combine over scales
            if concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:
        pass
    return multi_scale_interp

class Lplane_field(nn.Module):
    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        spatial_distortion: Optional[SpatialDistortion],
        ratio_activation: Callable,
        linear_decoder: bool,
        linear_decoder_layers: Optional[int],
        k:Optional[int],
        num_frames: int,
        hues:Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.time_interval = 2/num_frames
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config
        self.num_frames = num_frames
        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.ratio_activation = ratio_activation
        self.linear_decoder = linear_decoder
        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)
        log.info(f"Initialized model grids: {self.grids}")
        self.num_frames = num_frames
        self.original = nn.Parameter(360*hues,requires_grad=False)
        self.hues = nn.Parameter(360*hues,requires_grad=True) 
        
        # 3. Init decoder params
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        # 3. Init decoder network
        self.geo_feat_dim = 15
        self.in_dim_color = (
                self.direction_encoder.n_output_dims
                + self.feature_dim
        )
        self.SV_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=2, #output s,v
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
            
        )
        self.Hue_select = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=k, #output s,v
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        self.Intensity = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=1, #output s,v
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def get_hues(self,  timestamps, n_rays, n_samples):
        timestamps = timestamps.reshape(n_rays*n_samples)
        _idx = ((timestamps+1.0)/2)*(self.num_frames-1)
        _idx = _idx.view(-1,1)
        idx = _idx.to(torch.int32)
        ceil_idx = torch.ceil(_idx).to(torch.int32)
        dff = _idx-idx
    
        interp = (torch.index_select(self.hues, 0, idx[:,0])*(1-dff) + torch.index_select(self.hues, 0, ceil_idx[:,0])*(dff))
        
        return interp
    def shift_time(self, timestamps):
        forward_timestamps=timestamps + 2/(self.num_frames-1)
        forward_timestamps = torch.clamp(forward_timestamps,min=-1.0,max=1.0)
        backward_timestamps=timestamps - 2/(self.num_frames-1)
        backward_timestamps = torch.clamp(backward_timestamps,min=-1.0,max=1.0)

        return forward_timestamps, backward_timestamps
    def get_feature(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""

        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
            
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None: # not none [4096]
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        hues = self.get_hues(timestamps, n_rays, n_samples)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)

        return features, hues
    
    def only_get_feature(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):

        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
            
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None: # not none [4096]
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)

        return features
    def only_get_directions(self, pts,directions):
        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
       # hues = hues.view(-1,1, hues.shape[-1]).expand(pts.shape[0],pts.shape[1],hues.shape[-1]).reshape(-1, hues.shape[-1])
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)
        return encoded_directions
    def forward(self,
                pts: torch.Tensor,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        features, hues = self.get_feature(pts, timestamps)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
       # hues = hues.view(-1,1, hues.shape[-1]).expand(pts.shape[0],pts.shape[1],hues.shape[-1]).reshape(-1, hues.shape[-1])
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)
        if self.linear_decoder:
            color_features = [features]
        else:
            color_features = [encoded_directions, features]
        color_features = torch.cat(color_features, dim=-1)
        S, V = torch.split(torch.sigmoid(self.SV_net(color_features)).to(directions),[1,1],dim=-1)#.view(n_rays, n_samples, 3)
        H =  torch.sum(hues*torch.softmax(self.Hue_select(color_features),dim=-1),dim=-1,keepdim=True).to(directions)
        rgb = HSV2RGB(H,S,V).view(n_rays,n_samples,3)
        intensity = self.Intensity(color_features).view(n_rays,n_samples,1)
        light = rgb*intensity
        return {"l_color": light.nan_to_num(1e-6), 'intensity':intensity.nan_to_num(1e-6), 'hues':hues.view(n_rays,n_samples,hues.shape[-1]).nan_to_num(1e-6)}
    def forward_wo_il(self,
                pts: torch.Tensor,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        features, hues = self.get_feature(pts, timestamps)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
       # hues = hues.view(-1,1, hues.shape[-1]).expand(pts.shape[0],pts.shape[1],hues.shape[-1]).reshape(-1, hues.shape[-1])
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)
        if self.linear_decoder:
            color_features = [features]
        else:
            color_features = [encoded_directions, features]
        color_features = torch.cat(color_features, dim=-1)
        S, V = torch.split(torch.sigmoid(self.SV_net(color_features)).to(directions),[1,1],dim=-1)#.view(n_rays, n_samples, 3)
        H =  torch.sum(hues*torch.softmax(self.Hue_select(color_features),dim=-1),dim=-1,keepdim=True).to(directions)
        rgb = HSV2RGB(H,S,V).view(n_rays,n_samples,3)
        intensity = self.Intensity(color_features).view(n_rays,n_samples,1)

        light = rgb
        return {"l_color": light.nan_to_num(1e-6), 'intensity':intensity.nan_to_num(1e-6), 'hues':hues.view(n_rays,n_samples,hues.shape[-1]).nan_to_num(1e-6)}
    
    def forward_change_hue(self,
                    hue: float,
                    il_ratio: float,
                    pts: torch.Tensor,
                    directions: torch.Tensor,
                    timestamps: Optional[torch.Tensor] = None):
        features, hues = self.get_feature(pts, timestamps)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
       # hues = hues.view(-1,1, hues.shape[-1]).expand(pts.shape[0],pts.shape[1],hues.shape[-1]).reshape(-1, hues.shape[-1])
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)

        if self.linear_decoder:
            color_features = [features]
        else:
            color_features = [encoded_directions, features]
            

        color_features = torch.cat(color_features, dim=-1)
        S, V = torch.split(torch.sigmoid(self.SV_net(color_features)).to(directions),[1,1],dim=-1)#.view(n_rays, n_samples, 3)
        if hue != -1:
            H = torch.full_like(S,hue) #torch.sum(hues*torch.softmax(self.Hue_select(color_features),dim=-1),dim=-1,keepdim=True).to(directions)
        else:
            #H =  torch.sum(hues*torch.softmax(self.Hue_select(color_features),dim=-1),dim=-1,keepdim=True).to(directions)
            H = torch.ones_like(S)*hue
        rgb = HSV2RGB(H,S,V).view(n_rays,n_samples,3)
        light = il_ratio*rgb*(self.Intensity(color_features).view(n_rays,n_samples,1))
        return {"l_color": light.nan_to_num(1e-6)}
    
    def forward_il_map(self,
                    il_map: torch.Tensor,
                    pts: torch.Tensor,
                    directions: torch.Tensor,
                    timestamps: Optional[torch.Tensor] = None):
        features, hues = self.get_feature(pts, timestamps)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
        # hues = hues.view(-1,1, hues.shape[-1]).expand(pts.shape[0],pts.shape[1],hues.shape[-1]).reshape(-1, hues.shape[-1])
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)

        if self.linear_decoder:
            color_features = [features]
        else:
            color_features = [encoded_directions, features]
            

        color_features = torch.cat(color_features, dim=-1)
        S, V = torch.split(torch.sigmoid(self.SV_net(color_features)).to(directions),[1,1],dim=-1)#.view(n_rays, n_samples, 3)

        H =  torch.sum(hues*torch.softmax(self.Hue_select(color_features),dim=-1),dim=-1,keepdim=True).to(directions)

        rgb = HSV2RGB(H,S,V).view(n_rays,n_samples,3)
        il_map = il_map.view(-1, 1, 1).expand(pts.shape[0],pts.shape[1],1)#.view(n_rays,n_samples,1) directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3
        light = il_map*rgb*(self.Intensity(color_features).view(n_rays,n_samples,1))
        return {"l_color": light.nan_to_num(1e-6)}
    """
    Time-shift variants:
      1) Only hue candidates are shifted in time.
      2) Hue probabilities are also shifted in time.
      3) All components (S, V, and hue) are shifted -- full time shift.
    """
    def shift_time_opt1(self,
                    pts: torch.Tensor,
                    directions: torch.Tensor,
                    light_timestamps: Optional[torch.Tensor] = None,
                    obj_timestamps: Optional[torch.Tensor] = None):
        
        obj_features, obj_hues = self.get_feature(pts, obj_timestamps)
        light_features, light_hues = self.get_feature(pts, light_timestamps)

        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
       # hues = hues.view(-1,1, hues.shape[-1]).expand(pts.shape[0],pts.shape[1],hues.shape[-1]).reshape(-1, hues.shape[-1])
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)

        if self.linear_decoder:
            obj_color_features = [obj_features]
            light_color_features = [light_features]

        else:
            obj_color_features = [encoded_directions, obj_features]
            light_color_features = [encoded_directions, light_features]

        obj_color_features = torch.cat(obj_color_features, dim=-1)
        light_color_features = torch.cat(light_color_features, dim=-1)

        S, V = torch.split(torch.sigmoid(self.SV_net(obj_color_features)).to(directions),[1,1],dim=-1)#.view(n_rays, n_samples, 3)
        H =  torch.sum(light_hues*torch.softmax(self.Hue_select(obj_color_features),dim=-1),dim=-1,keepdim=True).to(directions)

        rgb = HSV2RGB(H,S,V).view(n_rays,n_samples,3)
        light = rgb*(self.Intensity(obj_color_features).view(n_rays,n_samples,1))
        return {"l_color": light.nan_to_num(1e-6)}
    
    def shift_time_opt2(self,
                    pts: torch.Tensor,
                    directions: torch.Tensor,
                    light_timestamps: Optional[torch.Tensor] = None,
                    obj_timestamps: Optional[torch.Tensor] = None):
        obj_features, obj_hues = self.get_feature(pts, obj_timestamps)
        light_features, light_hues = self.get_feature(pts, light_timestamps)

        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
       # hues = hues.view(-1,1, hues.shape[-1]).expand(pts.shape[0],pts.shape[1],hues.shape[-1]).reshape(-1, hues.shape[-1])
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)

        if self.linear_decoder:
            obj_color_features = [obj_features]
            light_color_features = [light_features]

        else:
            obj_color_features = [encoded_directions, obj_features]
            light_color_features = [encoded_directions, light_features]

        obj_color_features = torch.cat(obj_color_features, dim=-1)
        light_color_features = torch.cat(light_color_features, dim=-1)

        S, V = torch.split(torch.sigmoid(self.SV_net(obj_color_features)).to(directions),[1,1],dim=-1)#.view(n_rays, n_samples, 3)
        H =  torch.sum(light_hues*torch.softmax(self.Hue_select(light_color_features),dim=-1),dim=-1,keepdim=True).to(directions)

        rgb = HSV2RGB(H,S,V).view(n_rays,n_samples,3)
        light = rgb*(self.Intensity(obj_color_features).view(n_rays,n_samples,1))
        return {"l_color": light.nan_to_num(1e-6)}

    def shift_time_opt3(self,
                    pts: torch.Tensor,
                    directions: torch.Tensor,
                    light_timestamps: Optional[torch.Tensor] = None,
                    obj_timestamps: Optional[torch.Tensor] = None):
        obj_features, obj_hues = self.get_feature(pts, obj_timestamps)
        light_features, light_hues = self.get_feature(pts, light_timestamps)

        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
       # hues = hues.view(-1,1, hues.shape[-1]).expand(pts.shape[0],pts.shape[1],hues.shape[-1]).reshape(-1, hues.shape[-1])
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)

        if self.linear_decoder:
            obj_color_features = [obj_features]
            light_color_features = [light_features]

        else:
            obj_color_features = [encoded_directions, obj_features]
            light_color_features = [encoded_directions, light_features]

        obj_color_features = torch.cat(obj_color_features, dim=-1)
        light_color_features = torch.cat(light_color_features, dim=-1)

        S, V = torch.split(torch.sigmoid(self.SV_net(obj_color_features)).to(directions),[1,1],dim=-1)#.view(n_rays, n_samples, 3)
        H =  torch.sum(light_hues*torch.softmax(self.Hue_select(light_color_features),dim=-1),dim=-1,keepdim=True).to(directions)

        rgb = HSV2RGB(H,S,V).view(n_rays,n_samples,3)
        light = rgb*(self.Intensity(light_color_features).view(n_rays,n_samples,1))
        return {"l_color": light.nan_to_num(1e-6)}    
    def only_get_intensity(self,
                    pts: torch.Tensor,
                    directions: torch.Tensor,
                    timestamps: Optional[torch.Tensor] = None):
        n_rays, n_samples = pts.shape[:2]
        light_features = self.only_get_feature(pts,timestamps)
        if self.linear_decoder:
            light_color_features = [light_features]
        else:
            direction_features = self.only_get_directions(pts,directions)
            light_color_features = [direction_features, light_features]
        light_color_features = torch.cat(light_color_features,dim=-1)
        intensity = self.Intensity(light_color_features).view(n_rays,n_samples,1)
        return intensity
    
    def spatial_smoothness(self,
                    pts: torch.Tensor,
                    directions: torch.Tensor,
                    timestamps: Optional[torch.Tensor] = None):
        pts_jittor = pts + torch.randn_like(pts) * 0.01
        
        jittored_intensity =  self.only_get_intensity(pts_jittor,directions,timestamps)

        return jittored_intensity
    
    def temporal_smoothness(self,
                    pts: torch.Tensor,
                    directions: torch.Tensor,
                    timestamps: Optional[torch.Tensor] = None):
        forward_shift, backward_shift = self.shift_time(timestamps)
        forward_intensity = self.only_get_intensity(pts,directions,forward_shift)
        backward_intensity = self.only_get_intensity(pts,directions,backward_shift)

        return forward_intensity, backward_intensity
    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = [
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        nn_params.append(self.Hue_select.named_parameters(prefix="Hue_select"))
        nn_params.append(self.SV_net.named_parameters(prefix="SV_net"))
        nn_params.append(self.Intensity.named_parameters(prefix="Intensity"))

        hues = [self.hues]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys() and k != "hues"
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "hues" : hues,
            "other": list(other_params.values()),
            
        }
        

         