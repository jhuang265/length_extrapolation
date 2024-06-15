# Length Extrapolation Project

## Introduction

## Setup Environments

### RULER

For RULER, we can create a virtual environment as follows

```bash
# Load modules
module load anaconda/3
module load cuda/11.8
module load singularity

virtualenv --no-download $ENV_NAME
source $ENV_NAME/bin/activate

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install packaging ninja
pip install transformers accelerate tenacity flash-attn nltk
pip install 'causal-conv1d>=1.2.0' mamba-ssm
pip install install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

Alternatively, you can use the pre-built Dockerfile provided to build a container. If you decide to use this, note that some errors may occur.

## Running Experiments
