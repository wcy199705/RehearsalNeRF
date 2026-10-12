"""Base configuration for RehearsalNeRF (ReHeaRF).

All scene configs inherit from this base. Override only what differs per scene.

Usage in scene configs:
    from configs.base_config import base_config
    config = {**base_config, 'data_dirs': ['./dataset/KPOP/01'], 'logdir': './logs/01'}
"""

base_config = {
    'expname': 'RehearsalNeRF',
    'device': 'cuda:0',

    # Data settings
    'data_downsample': 2,
    'pose_type': 'reh',
    'contract': False,
    'ndc': True,
    'ndc_far': 2.6,
    'near_scaling': 0.95,
    'isg': False,
    'masked_weight': True,
    'isg_step': -1,
    'ist_step': 9000000000,
    'keyframes': False,
    'scene_bbox': [[-3.0, -1.8, -1.2], [3.0, 1.8, 1.2]],
    'use_mask': True,
    'num_frames': 75,

    # Optimization
    'num_steps': 120001,
    'batch_size': 3072,
    'scheduler_type': 'warmup_cosine',
    'optim_type': 'adam',
    'lr': 0.001,

    # Loss weights
    'loss_weights': {
        'illum_loss': 0.5,
        'reh_static_loss': 0.5,
        'liv_static_loss': 0.5,
        'liv_dynamic_loss': 0.5,
        'liv_wo_il_loss': 0.5,
        'wrong_weight_loss': 1e-2,
        'scale_loss': 1e-3,
        'alpha_loss': 1e-3,
        'dynamic_loss': 1e-3,
        'hue_loss': 0.0,
    },

    # Regularization
    'distortion_loss_weight': 0.001,
    'histogram_loss_weight': 1.0,
    'l1_time_planes': 0.0,
    'l1_time_planes_proposal_net': 0.0,
    'plane_tv_weight': 0.0002,
    'plane_tv_weight_proposal_net': 0.0002,
    'time_smoothness_weight': 0.0,
    'time_smoothness_weight_proposal_net': 0.0,
    'motion_loss_weight': 0.0,
    'big_motion_loss_weight': 0.0,
    'change_hue': 20000,

    # Training settings
    'save_every': 30000,
    'valid_every': 30000,
    'save_outputs': True,
    'train_fp16': True,
    'valid_idx': 0,

    # Raymarching
    'single_jitter': False,
    'num_samples': 96,
    'num_proposal_samples': [512, 256],
    'num_proposal_iterations': 2,
    'use_same_proposal_network': False,
    'use_proposal_weight_anneal': True,
    'proposal_net_args_list': [
        {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [128, 128, 128]},
        {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [256, 256, 256]},
    ],
    'multiscale_res': [1, 2, 4, 8],

    # Lightning Field (L) — illumination/hue
    'L_concat_features_across_scales': True,
    'L_density_activation': 'trunc_exp',
    'L_grid_config': [{
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 75],
    }],
    'L_appearance_embedding_dim': 32,

    # BRDF interaction decoder + tone mapping (from the original final config Change_dynamic/01.py).
    # These MUST be set: without them LowrankModel falls back to defaults (attention + 'rgb'),
    # whose brdf/tone-mapping path overflows in fp16 (alpha_loss=-log(brdf) -> NaN, ~step 700) and
    # in fp32 (~step 36.5k). The original 'mlp' + 'nerv' path is numerically stable.
    'intensity_decoder_type': 'mlp',       # mlp (stable) vs attention (diverges)
    'intensity_decoder_hidden_dim': 16,
    'intensity_decoder_out_dim': 3,
    'tone_mapping_option': 'nerv',         # nerv (stable) vs rgb (diverges)
    'tone_mapping_hidden_dim': 16,

    # Static Field (S) — base geometry
    'S_concat_features_across_scales': True,
    'S_density_activation': 'trunc_exp',
    'S_linear_decoder': False,
    'S_grid_config': [{
        'grid_dimensions': 2,
        'input_coordinate_dim': 3,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64],
    }],

    # Dynamic Field (D) — motion
    'D_concat_features_across_scales': True,
    'D_density_activation': 'trunc_exp',
    'D_linear_decoder': False,
    'D_grid_config': [{
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 75],
    }],
}
