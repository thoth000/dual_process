from omegaconf import OmegaConf, ListConfig, DictConfig
import os
import sys
import torch
from PIL import Image
import argparse
from pathlib import Path

from dual_process import dig_helpers


def init_config():
    """プロジェクト慣習に従った設定初期化"""
    global_config = OmegaConf.create({})
    for config_name in sys.argv[1].split(","):
        global_config = OmegaConf.merge(global_config, OmegaConf.load(config_name))
    if len(sys.argv) > 2:
        cli_overrides = OmegaConf.from_cli(sys.argv[2:])
        global_config = OmegaConf.merge(global_config, cli_overrides)
    OmegaConf.resolve(global_config)
    
    # 出力フォルダの設定
    save_folder = global_config.get("save_folder", "generated_with_lora")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    
    # 設定ファイルを保存
    OmegaConf.save(global_config, f"{save_folder}/generation_config.yaml")
    return global_config, save_folder


def load_lora_weights(pipe, lora_files, lora_weights=None, lora_config=None):
    """
    複数のLoRAファイルを読み込み、パイプラインに適用
    
    Args:
        pipe: 画像生成パイプライン
        lora_files: LoRAファイルのパスリスト
        lora_weights: 各LoRAの重みリスト（Noneの場合は全て1.0）
        lora_config: LoRA設定辞書
    """
    # ListConfig/DictConfigを通常のPythonオブジェクトに変換
    if isinstance(lora_files, (ListConfig, DictConfig)):
        lora_files = OmegaConf.to_container(lora_files, resolve=True)
    if isinstance(lora_weights, (ListConfig, DictConfig)):
        lora_weights = OmegaConf.to_container(lora_weights, resolve=True)
    
    # リストでない場合はリストに変換
    if not isinstance(lora_files, list):
        lora_files = [lora_files]
    
    if lora_weights is None:
        lora_weights = [1.0] * len(lora_files)
    elif not isinstance(lora_weights, list):
        lora_weights = [lora_weights] * len(lora_files)
    
    print(f"Loading {len(lora_files)} LoRA files...")
    
    # LoRAアダプターを作成（dig_pipeline.pyの慣習に従う）
    default_lora_kwargs = {
        "r": 16,
        "lora_lr": 5e-5,
        "init_lora_weights": "gaussian",
        "lora_dropout": 0.0,
        "target_modules": [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj"
        ]
    }
    
    if lora_config and "lora_kwargs" in lora_config:
        lora_kwargs = lora_config["lora_kwargs"]
        if isinstance(lora_kwargs, DictConfig):
            lora_kwargs = OmegaConf.to_container(lora_kwargs, resolve=True)
    else:
        lora_kwargs = default_lora_kwargs
    
    lora_params = dig_helpers.create_lora(pipe, **lora_kwargs)
    
    # 最初のLoRAファイルを読み込み（複数LoRAの場合は最初のもので初期化）
    if lora_files:
        lora_file = str(lora_files[0])
        
        if not os.path.exists(lora_file):
            print(f"Warning: Primary LoRA file not found: {lora_file}")
            return pipe
        
        print(f"Loading primary LoRA: {lora_file}")
        
        try:
            # 拡張子を除いたファイル名でload_weights()を呼び出し
            lora_file_path = lora_file.replace(".pt", "") if lora_file.endswith(".pt") else lora_file
            dig_helpers.load_weights(pipe, lora_file_path)
            
            print("LoRA loading completed!")
            print(f"LoRA will be applied with weight: {lora_weights[0]}")
            
            # パイプラインにLoRA重みを記録（生成時に使用）
            pipe._lora_weight = lora_weights[0]
            
            # デバッグ: LoRAアダプターの状態確認
            backbone = dig_helpers.get_backbone(pipe)
            if hasattr(backbone, "peft_config"):
                print(f"✅ LoRA adapter created: {list(backbone.peft_config.keys())}")
                print(f"✅ Active adapters: {backbone.active_adapters()}")
            else:
                print("❌ No LoRA adapter found!")
            
        except Exception as e:
            print(f"Warning: Failed to load LoRA {lora_file}: {e}")
            return pipe
    
    return pipe


def generate_images(pipe, config, save_folder):
    """画像生成を実行（LoRA適用版とオリジナル版の比較生成対応）"""
    generation_config = config.get("generation", {})
    
    # ListConfig/DictConfigを適切に処理
    if isinstance(generation_config, DictConfig):
        generation_config = OmegaConf.to_container(generation_config, resolve=True)
    
    prompts = generation_config.get("prompts", ["Photo of a man"])
    if isinstance(prompts, (ListConfig, DictConfig)):
        prompts = OmegaConf.to_container(prompts, resolve=True)
    elif isinstance(prompts, str):
        prompts = [prompts]
    
    num_images_per_prompt = generation_config.get("num_images_per_prompt", 4)
    
    # generator_kwargsを設定から取得
    generator_kwargs = config.get("generator_kwargs", {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 3.5
    })
    
    # DictConfigを通常の辞書に変換
    if isinstance(generator_kwargs, DictConfig):
        generator_kwargs = OmegaConf.to_container(generator_kwargs, resolve=True)
    
    # 比較生成の設定を取得
    lora_config = config.get("lora_generation", {})
    if isinstance(lora_config, DictConfig):
        lora_config = OmegaConf.to_container(lora_config, resolve=True)
    
    generate_original = lora_config.get("generate_original", False)
    original_folder = lora_config.get("original_folder", "original")
    lora_folder = lora_config.get("lora_folder", "with_lora")
    
    print(f"Generating images with {len(prompts)} prompts...")
    
    # デバッグ: LoRA状態確認
    backbone = dig_helpers.get_backbone(pipe)
    has_lora = hasattr(backbone, "peft_config")
    if has_lora:
        print(f"🔧 LoRA adapters available: {list(backbone.peft_config.keys())}")
        lora_weight = getattr(pipe, '_lora_weight', 1.0)
        print(f"🔧 LoRA weight will be: {lora_weight}")
    else:
        print("🔧 No LoRA adapters - using base model")
    
    if generate_original:
        print(f"📊 Comparison mode enabled:")
        print(f"  - Original images → {original_folder}/")
        print(f"  - LoRA images → {lora_folder}/")
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: '{prompt}'")
        
        # プロンプト用フォルダ作成
        prompt_folder = os.path.join(save_folder, f"prompt_{prompt_idx:03d}")
        os.makedirs(prompt_folder, exist_ok=True)
        
        # プロンプトをファイルに保存
        with open(os.path.join(prompt_folder, "prompt.txt"), "w") as f:
            f.write(prompt)
        
        # オリジナル画像用フォルダ（必要に応じて）
        if generate_original:
            original_subfolder = os.path.join(prompt_folder, original_folder)
            os.makedirs(original_subfolder, exist_ok=True)
        
        # LoRA画像用フォルダ（LoRAが利用可能な場合）
        if has_lora:
            lora_subfolder = os.path.join(prompt_folder, lora_folder)
            os.makedirs(lora_subfolder, exist_ok=True)
        
        for img_idx in range(num_images_per_prompt):
            # シード設定
            seed = generation_config.get("seed", 42) + img_idx
            
            # 1. オリジナル画像生成（LoRAなし）
            if generate_original:
                print(f"  Generating original image {img_idx + 1}/{num_images_per_prompt}...")
                try:
                    generator = torch.Generator(device=pipe.device).manual_seed(seed)
                    
                    with torch.no_grad():
                        # LoRAなしで生成（weight=0）
                        if has_lora:
                            with dig_helpers.LoraManager(pipe, enter_weights=0):
                                result = pipe(
                                    prompt=prompt,
                                    generator=generator,
                                    **generator_kwargs
                                )
                                original_image = result.images[0]
                        else:
                            result = pipe(
                                prompt=prompt,
                                generator=generator,
                                **generator_kwargs
                            )
                            original_image = result.images[0]
                    
                    # オリジナル画像保存
                    original_path = os.path.join(prompt_folder, original_folder, f"original_{img_idx:03d}_seed_{seed}.png")
                    original_image.save(original_path)
                    print(f"    Saved original: {os.path.basename(original_path)}")
                    
                except Exception as e:
                    print(f"    Error generating original image {img_idx}: {e}")
                    continue
            
            # 2. LoRA適用画像生成
            if has_lora:
                print(f"  Generating LoRA image {img_idx + 1}/{num_images_per_prompt}...")
                try:
                    generator = torch.Generator(device=pipe.device).manual_seed(seed)
                    
                    with torch.no_grad():
                        # LoRAManagerを使ってLoRA重みを適用
                        lora_weight = getattr(pipe, '_lora_weight', 1.0)
                        
                        with dig_helpers.LoraManager(pipe, enter_weights=lora_weight):
                            result = pipe(
                                prompt=prompt,
                                generator=generator,
                                **generator_kwargs
                            )
                            lora_image = result.images[0]
                    
                    # LoRA画像保存
                    lora_path = os.path.join(prompt_folder, lora_folder, f"lora_{img_idx:03d}_seed_{seed}.png")
                    lora_image.save(lora_path)
                    print(f"    Saved LoRA: {os.path.basename(lora_path)}")
                    
                except Exception as e:
                    print(f"    Error generating LoRA image {img_idx}: {e}")
                    continue
            else:
                # LoRAがない場合は通常の画像生成
                print(f"  Generating image {img_idx + 1}/{num_images_per_prompt} (no LoRA)...")
                try:
                    generator = torch.Generator(device=pipe.device).manual_seed(seed)
                    
                    with torch.no_grad():
                        result = pipe(
                            prompt=prompt,
                            generator=generator,
                            **generator_kwargs
                        )
                        image = result.images[0]
                    
                    # 通常画像保存
                    image_path = os.path.join(prompt_folder, f"image_{img_idx:03d}_seed_{seed}.png")
                    image.save(image_path)
                    print(f"    Saved: {os.path.basename(image_path)}")
                    
                except Exception as e:
                    print(f"    Error generating image {img_idx}: {e}")
                    continue
    
    print(f"\nGeneration completed! Results saved to: {save_folder}")
    if generate_original and has_lora:
        print(f"📊 Comparison images generated:")
        print(f"  - Original images (LoRA weight=0): {original_folder}/")
        print(f"  - LoRA images (LoRA weight={getattr(pipe, '_lora_weight', 1.0)}): {lora_folder}/")


def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("Usage: python generate_with_lora.py <config_files> [overrides...]")
        print("Example: python generate_with_lora.py configs/base.yaml,configs/pipe/schnell.yaml,configs/generation/my_lora.yaml")
        print("         python generate_with_lora.py configs/base.yaml,configs/pipe/schnell.yaml save_folder=my_output")
        sys.exit(1)
    
    # 設定初期化
    config, save_folder = init_config()
    
    # パイプライン読み込み
    print("Loading pipeline...")
    pipe = dig_helpers.load_pipe(**config["pipe_kwargs"])
    
    # LoRAファイルを読み込み
    lora_config = config.get("lora_generation", {})
    if isinstance(lora_config, DictConfig):
        lora_config = OmegaConf.to_container(lora_config, resolve=True)
    
    lora_files = lora_config.get("lora_files", [])
    lora_weights = lora_config.get("lora_weights", None)
    
    if lora_files:
        pipe = load_lora_weights(pipe, lora_files, lora_weights, config)
    else:
        print("No LoRA files specified. Generating with base model.")
    
    # 画像生成
    generate_images(pipe, config, save_folder)


if __name__ == "__main__":
    main()
