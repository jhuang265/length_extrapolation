# Length Extrapolation Project

## Introduction

## Setup Environments

### RULER

#### Using an environment directly.

For RULER, we can create a virtual environment as follows

```bash
# Load modules
module load anaconda/3
module load cuda/12.2

virtualenv --no-download $ENV_NAME
source $ENV_NAME/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install packaging ninja
pip install transformers accelerate tenacity flash-attn nltk
pip install 'causal-conv1d>=1.2.0' mamba-ssm
pip install install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

Occasionally you may encounter an error based on your system and therefore might need to run the following:
```bash
MAX_JOBS=1 pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.1.0/mamba_ssm-2.1.0+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.3.0.post1/causal_conv1d-1.3.0.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

Alternatively, you can use the pre-built Dockerfile provided to build a container. If you decide to use this, note that some errors may occur.

#### Using a container

The original authors of RULER provide a Dockerfile as well as a pre-built image for you to use.

In the case of the image, first run the container in writable model (in Singularity, this done with a `singularity shell --writable $IMAGE_PATH`) and then update `mamba-ssm` so that it works with the `Mamba2` models.

Additionally, you may need to update `causal-conv1d` if you observe a symbol issue with CUDA.

## Running Experiments

### RULER

All the files for running RULER experiments are available in the `RULER` folder (you must first pull the git submodule).

There will be a `scripts/` folder which contains the code. From this folder, you should run jobs directly from `mila_run_scripts`.
