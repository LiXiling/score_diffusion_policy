ENVNAME="score-diff-policy"

conda create --name $ENVNAME python=3.11 &&
eval "$(conda shell.bash hook)" &&
conda activate $ENVNAME &&

# Order of Operations is important!
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y &&
#conda install -c conda-forge torchsde -y &&
conda install -c conda-forge dill wandb tqdm einops zarr pandas numba -y &&
conda install -c conda-forge shapely scikit-image matplotlib opencv -y &&
conda install -c conda-forge diffusers gymnasium pygame pymunk -y &&
conda install -c conda-forge av -y &&

# Install Last - Somehow always overwritten
conda install -c conda-forge hydra-core -y &&

conda install black isort pre-commit -c conda-forge &&

pip install torchsde torchdiffeq  opencv-python&&
pip install -e .
