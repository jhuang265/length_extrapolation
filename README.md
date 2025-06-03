# Length Extrapolation Project

## Introduction

This is the code for the COLING 2025 paper "How Well Can a Long Sequence Model Model Long Sequences? Comparing Architectural Inductive Biases on Long-Context Abilities".

Everything is actually based on the [RULER](https://github.com/NVIDIA/RULER) repository, which was modified to support some additional experiments.

## Setup Environments

Note that everything was run in a SLURM cluster. The following are the instructions I ran but which may be different for you.

### RULER

#### Using an environment directly.

For RULER, we can create a virtual environment as follows

```bash
# Load modules
module load anaconda/3
module load cuda/11.8

virtualenv --no-download $ENV_NAME
source $ENV_NAME/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Make your you install the correct version based on CUDA.

pip install packaging ninja

pip install transformers accelerate tenacity flash-attn nltk

pip install 'causal-conv1d>=1.2.0' mamba-ssm # You may need to manually install wheels.

pip install install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

Sometimes you may need to use a `MAX_JOBS=1` before the `pip` command. Once everything is ready your environment should be suited to run all code without issue.

#### Using a container

The original authors of RULER provide a Dockerfile as well as a pre-built image (`cphsieh/ruler:0.1.0`) for you to use. Use whatever way to create a new container, however preferably with Singularity, but make the container writable as there may be packages for you to change. For example

```bash
singularity build --sandbox ruler/ docker://cphsieh/ruler:0.1.0
```

In the case of the image, first run the container in writable model (in Singularity, this done with a `singularity shell --writable $IMAGE_PATH`) and then update `mamba-ssm` so that it works with the `Mamba2` models.

Additionally, you may need to update `causal-conv1d` if you observe a symbol issue with CUDA.

Once all your packages are in order, you can build the container directly from the writable folder
```bash
singularity build ruler.sif ruler/
```
and launch it using
```bash
singularity exec --nv --cleanenv [ARGS]
```
or
```bash
singularity shell --nv --cleanenv [ARGS]
```

#### Common issues

There are some common issues you might observe. One of them is a logits issue with Mamba models. Most of the time, this can be fixed by finding the file `mamba/mamba_ssm/utils/generation.py` and changing the following in the `sample` function (as long as you never changed the `temperature` argument originally).
```python
if temperature != 1.0:
    logits_top /= temperature
```

In particular, you should add a check to make sure that temperature is defined and greater than 0, such as
```python
if temperature != 1.0 and temperature > 0:
    logits_top /= temperature
```

## Running Experiments

### RULER

All the files for running RULER experiments are available in the `RULER` folder (you must first pull the git submodule).

There will be a `scripts/` folder which contains the code. From this folder, you should run jobs directly from `mila_run_scripts`.

## Citation

```
@inproceedings{huang-2025-well,
    title = "How Well Can a Long Sequence Model Model Long Sequences? Comparing Architectural Inductive Biases on Long-Context Abilities",
    author = "Huang, Jerry",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.3/",
    pages = "29--39"
}
```
