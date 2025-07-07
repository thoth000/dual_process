pipe=sana
generation_config=lora_demo
lora_name=man-smile
experiment_name=demo_sana_idefics2

# 出力フォルダ
save_folder=lora_generated/${generation_config}_${pipe}

python3 generate_with_lora.py configs/base.yaml,configs/pipe/${pipe}.yaml,configs/generation/${generation_config}.yaml save_folder=${save_folder}
