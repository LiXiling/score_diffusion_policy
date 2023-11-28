ENVNAME="score-diff-policy"

conda create --name $ENVNAME python=3.11
eval "$(conda shell.bash hook)"
conda activate $ENVNAME

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda update -c conda-forge hydra-core
conda install -c conda-forge dill wandb tqdm einops zarr pandas numba -y
conda install -c conda-forge diffusers gymnasium


conda install lightning scipy matplotlib -c conda-forge

conda install black isort pre-commit -c conda-forge

pip install lightning-bolts
pip install -e .
