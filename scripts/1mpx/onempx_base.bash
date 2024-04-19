#!/usr/bin/env bash

source activate events_signals

python RVT/train.py model=rnndet dataset=gen4 dataset.path=/shares/rpg.ifi.uzh/nzubic/datasets/RVT/gen4_new_no_psee_filter wandb.project_name=ssms_event_cameras \
wandb.group_name=1mpx +experiment/gen4=base.yaml hardware.gpus=[0,1] batch_size.train=6 batch_size.eval=6 \
hardware.num_workers.train=12 hardware.num_workers.eval=4
