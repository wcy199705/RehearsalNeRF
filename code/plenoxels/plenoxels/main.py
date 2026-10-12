import argparse
import ast
import importlib.util
import logging
import os
import pprint
import sys
from typing import List, Dict, Any

import numpy as np
import torch
import torch.utils.data

from runners import kpop_trainer
from utils.create_rendering import (
    render_to_path, decompose_space_time, rendering_fix_dynamic,
    rendering_fix_light, hue_fix, hue_change, no_light,
    custom_hue, custom_hue_single_Img, custom_single,
)
from utils.parse_args import parse_optfloat


log = logging.getLogger(__name__)


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers,
                        force=True)


def load_data(data_downsample, data_dirs, validate_only: bool, render_only: bool,
              application_only: bool, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)
    return kpop_trainer.load_data(
        data_downsample, data_dirs, validate_only=validate_only,
        application_only=application_only, render_only=render_only, **kwargs)


def init_trainer(**kwargs):
    return kpop_trainer.KpopTrainer(**kwargs)


def save_config(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
        for key in config.keys():
            f.write("%s\t%s\n" % (key, config[key]))


def parse_override_value(value: str) -> Any:
    """Parse a command-line override value as a Python literal (int, float, list, ...),
    falling back to the raw string (e.g. paths)."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def main():
    setup_logging()

    p = argparse.ArgumentParser(description="RehearsalNeRF Training and Rendering")

    # Rendering modes
    p.add_argument('--render-only', action='store_true')
    p.add_argument('--render_fig8', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--hue_test', action='store_true')
    p.add_argument('--frozen_light', action='store_true')
    p.add_argument('--frozen_motion', action='store_true')
    p.add_argument('--render_wo_il', action='store_true')

    # Intensity map rendering
    p.add_argument('--intensity_left', action='store_true')
    p.add_argument('--intensity_right', action='store_true')
    p.add_argument('--intensity_center', action='store_true')
    p.add_argument('--intensity_square', action='store_true')
    p.add_argument('--intensity_lr', action='store_true')
    p.add_argument('--intensity_map', type=str, default=None,
                   help='Path to intensity map .pth file')

    # Application modes
    p.add_argument('--app_test_ill', action='store_true')
    p.add_argument('--app_brighter', action='store_true')
    p.add_argument('--decomp_rendering_only', action='store_true')
    p.add_argument('--applications_only', action='store_true')
    p.add_argument('--change_hue_only', action='store_true')
    p.add_argument('--single_app_only', action='store_true')
    p.add_argument('--spacetime-only', action='store_true')
    p.add_argument('--hue_video_only', action='store_true')

    # Time shift experiments
    p.add_argument('--rebuttal_shift_time_5', action='store_true')
    p.add_argument('--rebuttal_shift_time_10', action='store_true')
    p.add_argument('--rebuttal_shift_time_20', action='store_true')

    # Config and checkpoints
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--model_name', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('override', nargs=argparse.REMAINDER)

    args = p.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import config
    spec = importlib.util.spec_from_file_location(
        os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config

    # Process overrides from command line as key=value pairs
    overrides: List[str] = args.override
    overrides_dict = {key: parse_override_value(value)
                      for key, value in (ovr.split("=", 1) for ovr in overrides)}
    config.update(overrides_dict)

    if "num_frames" not in config:
        raise ValueError("Config must contain 'num_frames' for kpop dataset.")

    validate_only = args.validate_only
    render_only = args.render_only
    spacetime_only = args.spacetime_only

    if validate_only and render_only:
        raise ValueError("render_only and validate_only are mutually exclusive.")
    if render_only and spacetime_only:
        raise ValueError("render_only and spacetime_only are mutually exclusive.")
    if validate_only and spacetime_only:
        raise ValueError("validate_only and spacetime_only are mutually exclusive.")

    pprint.pprint(config)
    if validate_only or render_only:
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
    else:
        save_config(config)

    # Determine if we need eval-mode data loading
    eval_mode = (validate_only or args.render_fig8 or args.hue_test
                 or args.frozen_motion or args.frozen_light
                 or args.intensity_center or args.intensity_right
                 or args.intensity_left or args.intensity_square
                 or args.intensity_lr or args.app_test_ill
                 or args.app_brighter or args.render_wo_il)

    data = load_data(
        validate_only=eval_mode,
        render_only=render_only or spacetime_only,
        application_only=args.applications_only,
        **config)
    config.update(data)
    trainer = init_trainer(**config)

    if args.log_dir is not None:
        checkpoint_path = os.path.join(args.log_dir, args.model_name)
        training_needed = not (validate_only or render_only or spacetime_only)
        trainer.load_model(torch.load(checkpoint_path), training_needed=training_needed)
        log.info("Checkpoint loaded from %s", checkpoint_path)

    # Dispatch to appropriate mode
    if args.render_wo_il:
        trainer.render_wo_il_version()
    elif args.render_fig8:
        trainer.render_fig8()
    elif validate_only:
        trainer.validate()
    elif render_only:
        render_to_path(trainer, extra_name="")
    elif args.hue_test:
        for hue in [0, 5, 30, 180, 240, 250, 315]:
            trainer.supp_multiple_hues(hue_value=hue)
    elif args.frozen_motion:
        for t in [0, 10, 20, 30, 40, 50]:
            trainer.supp_frozen_motion(t)
    elif args.frozen_light:
        for t in [0, 10, 20, 30, 40, 50]:
            trainer.supp_frozen_light(t)
    elif args.intensity_left or args.intensity_right or args.intensity_center \
            or args.intensity_square or args.intensity_lr:
        if args.intensity_map is None:
            raise ValueError("--intensity_map <path.pth> is required for intensity modes.")
        il_map = torch.load(args.intensity_map)
        label = ('left' if args.intensity_left else
                 'right' if args.intensity_right else
                 'center' if args.intensity_center else
                 'square' if args.intensity_square else 'lr')
        trainer.supp_ill_map_change(il_map, label)
    elif args.app_test_ill:
        trainer.app_test_ill()
    elif args.app_brighter:
        trainer.app_test_brighter()
    elif args.decomp_rendering_only:
        rendering_fix_dynamic(trainer, dynamic_time=25)
    elif args.applications_only:
        from tqdm import trange
        for scene in [1, 40, 54]:
            hue_fix(trainer, hue_time=scene)
            hue_change(trainer, human_time=scene)
        no_light(trainer)
    elif args.hue_video_only:
        from tqdm import trange
        for i in trange(0, 12):
            custom_hue(trainer, hue=i * 30)
    elif args.single_app_only:
        for scene in [45]:
            for t in range(0, 75):
                custom_single(trainer, human_time=t, hue_time=scene)
    elif args.rebuttal_shift_time_5:
        for t in range(0, 75):
            custom_single(trainer, human_time=t, hue_time=(t + 5) % 75)
    elif args.rebuttal_shift_time_10:
        for t in range(0, 75):
            custom_single(trainer, human_time=t, hue_time=(t + 10) % 75)
    elif args.rebuttal_shift_time_20:
        for t in range(0, 75):
            custom_single(trainer, human_time=t, hue_time=(t + 20) % 75)
    elif spacetime_only:
        decompose_space_time(trainer, extra_name="")
    elif args.change_hue_only:
        for i in range(0, 12):
            for j in range(0, 5):
                custom_hue_single_Img(trainer, hue=i * 30, fix_time=j * 15)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
