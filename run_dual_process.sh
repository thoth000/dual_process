#!/bin/bash

pipe=sana
vlm=idefics2
dataset=five_apple

save_folder=runs/${dataset}_${pipe}_${vlm}
python3 script_dual_process.py configs/base.yaml,configs/pipe/${pipe}.yaml,configs/vlm/${vlm}.yaml,configs/dataset/${dataset}.yaml save_folder=${save_folder}