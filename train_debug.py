"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config')),
    # config_name='train_beso_transformer_lowdim_pusht_workspace.yaml'
    # config_name='train_diffusion_transformer_lowdim_pusht_workspace.yaml'
    # config_name='train_beso_transformer_hybrid_workspace.yaml'
    # config_name='train_beso_gpt_lowdim_pusht_workspace.yaml'
    # config_name='train_beso_mlp_lowdim_pusht_workspace.yaml'
    config_name='train_gpt_transformer_lowdim_pusht_workspace.yaml'
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    # cfg = OmegaConf.load(cfg)
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    
    # cfg_path = '/home/moritz/code/score_diffusion_policy/diffusion_policy/config/train_beso_transformer_lowdim_pusht_workspace.yaml'
    main()
