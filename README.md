# MARK
PyTorch implementation for "MARK: Multi-agent Collaboration with Ranking Guidance for Text-attributed Graph Clustering"

## Overview

In this paper, we introduce a new perspective of leveraging large language models (LLMs) to enhance text-attributed graph clustering and develop a novel approach named \underline{M}ulti-\underline{a}gent Collabo\underline{r}ation with Ran\underline{k}ing Guidance (\method{}). The core of our \method{} is to generate reliable guidance using the collaboration of three LLM-based agents as ranking-based supervision signals. In particular, we first conduct the coarse graph clustering, and utilize a concept agent to induce the semantics of each cluster. Then, we infer the robustness under perturbations to identify uncertain nodes and use a generation agent to produce synthetic text that closely aligns with their topology. An inference agent is adopted to provide the ranking semantics for each uncertain node in comparison to its synthetic counterpart. The consistent feedback between uncertain and synthetic texts is identified as reliable guidance for fine-tuning the clustering model within a ranking-based supervision objective. Experimental results on various benchmark datasets validate the effectiveness of the proposed \method{} compared with competing baselines.

<img src="ProteinGGT.png">

## Installation

Start by following this source codes:
```bash
git clone https://github.com/fuyw-aisw/MARK.git
cd ProteinScore
conda env create -f environment.yml
## or install the following dependencies
## step1: install PyTorch’s CUDA support on Linux
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
## step2: install pyg package
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.7.1%2Bcu110.html ### GPU
## step2: or install by the following source codes
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_cluster-1.5.8-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch_geometric==1.6.3
```
