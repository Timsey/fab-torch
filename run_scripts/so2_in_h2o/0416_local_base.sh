#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate fab

nvidia-smi

CUDA_VISIBLE_DEVICES=0 /home/timsey/anaconda3/envs/fab/bin/python \
  /home/timsey/Projects/fab-torch/experiments/solvation/run.py --config-name so2inh2o_forwardkl.yaml