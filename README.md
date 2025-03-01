# Score Entropy Discrete Diffusion
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo contains a PyTorch implementation for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
](https://arxiv.org/abs/2310.16834) by [Aaron Lou](https://aaronlou.com), [Chenlin Meng](https://cs.stanford.edu/~chenlin/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/).

![cover](assets/main.gif)

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

Simply run

```
conda env create -f environment.yml
```

which will create a ```sedd``` environment with packages installed. Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. The biggest factor is making sure that the ```torch``` and ```flash-attn``` packages use the same CUDA version (more found [here](https://github.com/Dao-AILab/flash-attention)).

## Working with Pretrained Models

### Download Models

Our pretrained models are hosted on huggingface ([small](https://huggingface.co/louaaron/sedd-small), [medium](https://huggingface.co/louaaron/sedd-medium)). However, models can also be loaded in locally (say after training). All functionality is found in ```load_model.py```.

```
# load in a pretrained model
pretrained_small_model, graph, noise = load_model("louaaron/sedd-small")
pretrained_medium_model, graph, noise = load_model("louaaron/sedd-medium")
# load in a local experiment
local_model, graph, noise = load_model("exp_local/experiment)
```

This loading gives the model, as well as the graph and noise (which are used for the loss/sampling setup).

### Run Sampling

We can run sampling using a command 

```
python run_sample.py --model_path MODEL_PATH --steps STEPS
```

We can also sample conditionally using

```
python run_sample_cond.py --model_path MODEL_PATH --step STEPS --prefix PREFIX --suffix SUFFIX
```

## Training New Models

### Run Training

We provide training code, which can be run with the command
```
python run_train.py
```
This creates a new directory `direc=exp_local/DATE/TIME` with the following structure (compatible with running sampling experiments locally)
```
├── direc
│   ├── .hydra
│   │   ├── config.yaml
│   │   ├── ...
│   ├── checkpoints
│   │   ├── checkpoint_*.pth
│   ├── checkpoints-meta
│   │   ├── checkpoint.pth
│   ├── samples
│   │   ├── iter_*
│   │   │   ├── sample_*.txt
│   ├── logs
```
Here, `checkpoints-meta` is used for reloading the run following interruptions, `samples` contains generated images as the run progresses, and `logs` contains the run output. Arguments can be added with `ARG_NAME=ARG_VALUE`, with important ones being:
```
ngpus                     the number of gpus to use in training (using pytorch DDP)
training.accum            number of accumulation steps, set to 1 for small and 2 for medium (assuming an 8x80GB node)
noise.type                one of geometric, loglinear 
graph.type                one of uniform, absorb
model                     one of small, medium
model.scale_by_sigma      set to False if graph.type=uniform (not yet configured)
```
Some example commands include
```
# training hyperparameters for SEDD absorb
python train.py noise_lib=loglinear graph.type=absorb model=medium training.accum=2
# training hyperparameters for SEDD uniform
python train.py noise_lib=geometric graph.type=uniform model=small model.scale_by_sigma=False
```

## Other Features

### SLURM compatibility

To train on slurm, simply run 
```
python train.py -m args
```

## Citation
```
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```
## Acknowledgements

This repository builds heavily off of [score sde](https://github.com/yang-song/score_sde_pytorch), [plaid](https://github.com/igul222/plaid), and [DiT](https://github.com/facebookresearch/DiT).


# Diffusion of Thoughts

<p align = "center">
<img src="fig/logo.png" width="100%" alt="logo" align=center />
</p>

<p align = "center">
<img src="fig/sample3_chain.gif" width="75%" alt="examples" align=center loop=infinite/>
</p>

This repository contains code for training and evaluating the models in the paper [Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models](https://arxiv.org/abs/2402.07754).

- Diffusion models have gained attention in text processing, offering many potential advantages
over traditional autoregressive models. We explore the integration of diffusion models and Chain-of-Thought (CoT), a well-established technique to improve the reasoning ability in autoregressive language models. 
- We propose Diffusion-of-Thought (DoT), allowing reasoning steps to diffuse over time through the diffusion process. In contrast to traditional autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT offers more flexibility in the trade-off between computation and reasoning performance. 
- Additionally, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning capabilities in diffusion language models.

<p align = "center">
<img src="fig/pipeline.png" width="95%" alt="pipeline" align=center />
</p>
<p align = "center">
DoT pipeline demonstration.
</p>

Our implementation of DoT is mainly based on [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) (*DiffuSeq: Sequence to Sequence Text Generation With Diffusion Models*) and [Plaid](https://github.com/igul222/plaid) (*Likelihood-Based Diffusion Language Models*). Our tasks and dataset configurations primarily follow the guidelines set by [Implicit CoT](https://github.com/da03/implicit_chain_of_thought). Thanks for these excellent work!

## Setup
All required packages can be found in requirements.txt. You can install them in a new environment with
```
conda create -n dot python=3.10
conda activate dot

git clone git@github.com:HKUNLP/diffusion-of-thoughts.git

# The following line to be replaced depending on your cuda version.
cd diffusion-of-thoughts
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

Install NVIDIA Apex with fused kernels:
```
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

```

## Finetuning from Plaid 1B
First download the weights from here: [Plaid 1B Weights Download Page](https://github.com/igul222/plaid/releases/tag/v1.0.0). Download data from here: [4by4/5by4/GSM8k-Aug data](https://github.com/da03/implicit_chain_of_thought/tree/main/data) and put them in the ./data folder with names 4by4/5by5/gsm8k.

Extract them:
```
cat plaid1b_weights.tar.gz.* | tar xvzf -
```

Then run the following code:

```
# DoT
python train.py --digit --fix_src --dataset gsm8k --steps 120000 --weights_path plaid1b_weights 

# DoT-MP
python train.py --digit --fix_src --cot --dataset gsm8k --steps 31000 --weights_path plaid1b_weights 
```

Please refer to `run_train.sh` for more training commands.


## Evaluation
Here are some commands for evaluation and please refer to `run_eval.sh` for more examples.
```
# dot (T=64 by default)
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-digit-steps120000 --fix_src --digit --dataset gsm8k --score_temp 0.5

# dot dpmsolver 
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-digit-steps120000 --fix_src --digit --dataset gsm8k --dpm_solver

# dot T=8
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-digit-steps120000 --fix_src --digit --dataset gsm8k --score_temp 0.5 --sampling_timesteps 8

# mp-dot
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --score_temp 0.5

```


Pretrained checkpoints are under policy checking and we will release them as soon as possible...

## Finetuning from SEDD
### Environment Setup
### 🐳 Docker

We recommend to setup the environment with our docker [image](https://hub.docker.com/r/simonlemaqwq/sedd), which will prepare the whole environment and ease your reproduction with minimal effort.

```bash
# Pull the image for finetuning on LLaMA2 from dockerhub
docker pull simonlemaqwq/sedd:v1.0

# Start the container, remember to replace <PROJECT_DIR> with your own project directory
docker run \
    --name sedd \
    --gpus all \
    --network=host \
    -v <PROJECT_DIR>:/workspace \
    -it simonlemaqwq/sedd:v1.0 /bin/bash

# Switch to the branch for Diffusion of Thoughts - SEDD
cd /workspace
git checkout dot-sedd
```


### 🐍 Conda
If you use the above docker image, this step can be skipped, because the conda env has been well prepared in it.
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate sedd

# Install required dependencies
pip install -r requirements.txt
```


### Finetune
All configurations for finetuning should be set in `configs/config.yaml`.

For `DoT`, set the config as follows:
```yaml
data:
  multipass: False
  hidden_thought: False
```

For `DoT-MP`, set the config as follows:
```yaml
data:
  multipass: True
  hidden_thought: False
```

Then simply run:
```bash
python finetune.py
```

### Inference
Specify the `--model_path` in the script before sampling.
```bash
# e.g. --model_path /workspace/exp_local/gsm8k/<date>/<ckpt_dir>

# For DoT:
bash run_sample_cond.sh

# For DoT-MP
bash run_sample_cond_mp.sh
```

### Evaluation
```bash
python cal_acc_gsm.py
```


## More Cases
<p align = "center">
<img src="fig/sample1_chain.gif" width="75%" alt="example1" align=center loop=infinite/>
</p>
<p align = "center">
<img src="fig/sample2_chain.gif" width="75%" alt="example2" align=center loop=infinite/>
</p>
<p align = "center">
<img src="fig/sample3_chain.gif" width="75%" alt="example3" align=center loop=infinite/>
</p>

## Citation
```
@article{ye2024diffusion,
  title={Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models},
  author={Ye, Jiacheng and Gong, Shansan and Chen, Liheng and Zheng, Lin and Gao, Jiahui and Shi, Han and Wu, Chuan and Li, Zhenguo and Bi, Wei and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2402.07754},
  year={2024}
}
```