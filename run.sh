#!/usr/bin/env bash

#Run balanced, distributed-feedback dynamic selection for an M-node network, with a given target rate and no noise bursts
python -u Code/dynamic_noise.py -v --M 4 --name "Default_Dynamic" --target 0.75 --noise_prob 0.0 --seed 1 -balanced -enable_DSF --distributionmode "Distributed-Feedback"

#Run balanced greedy CMI for an M-node network, with a given target rate, K iterations and no noise bursts
python -u Code/train_cmi_finetune.py -v --K 3 --M 4 --name "Default_CMI" --target 0.75 --noise_prob 0.0 --seed 1 -balanced -enable_DSF
