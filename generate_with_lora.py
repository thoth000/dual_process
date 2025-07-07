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
    複数のLoRAファイルを重み付きで加算し、パイプラインに適用
    
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
    
    # 重みリストの長さを調整
    if len(lora_weights) < len(lora_files):
        lora_weights.extend([1.0] * (len(lora_files) - len(lora_weights)))
    elif len(lora_weights) > len(lora_files):
        lora_weights = lora_weights[:len(lora_files)]
    
    print(f"Loading and combining {len(lora_files)} LoRA files with weights: {lora_weights}")
    
    # LoRAアダプターを作成
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
    
    # LoRAアダプターを作成
    lora_params = dig_helpers.create_lora(pipe, **lora_kwargs)
    
    # 複数LoRAファイルを重み付きで加算
    if lora_files:
        combined_state_dict = {}
        
        for i, (lora_file, weight) in enumerate(zip(lora_files, lora_weights)):
            lora_file = str(lora_file)
            
            if not os.path.exists(lora_file):
                print(f"Warning: LoRA file not found: {lora_file}")
                continue
            
            print(f"Loading LoRA {i+1}/{len(lora_files)}: {lora_file} (weight: {weight})")
            
            try:
                # LoRA重みを読み込み
                lora_state = torch.load(lora_file, map_location=pipe.device)
                
                # 重み付きで加算
                for param_name, param_tensor in lora_state.items():
                    if param_name not in combined_state_dict:
                        # 最初のLoRAの場合は重み付きでコピー
                        combined_state_dict[param_name] = weight * param_tensor.to(pipe.device)
                    else:
                        # 2番目以降のLoRAは重み付きで加算
                        combined_state_dict[param_name] += weight * param_tensor.to(pipe.device)
                
                print(f"  ✅ Added LoRA {i+1} with weight {weight}")
                
            except Exception as e:
                print(f"  ❌ Failed to load LoRA {lora_file}: {e}")
                continue
        
        # 加算された重みをパイプラインに適用
        if combined_state_dict:
            try:
                # dig_helpersの方法で適用
                backbone = dig_helpers.get_backbone(pipe)
                
                # 重みキーを適切な形式に変換
                formatted_state_dict = {}
                for name, param in combined_state_dict.items():
                    # .weight を .default.weight に変換
                    formatted_name = name.replace(".weight", ".default.weight")
                    formatted_state_dict[formatted_name] = param
                
                # state_dictを読み込み
                backbone.load_state_dict(formatted_state_dict, strict=False)
                
                print(f"✅ Successfully combined and loaded {len(combined_state_dict)} LoRA parameters")
                print(f"📊 Combined LoRA weights: {dict(zip(lora_files, lora_weights))}")
                
                # パイプラインにLoRA適用フラグを記録
                pipe._lora_weight = 1.0  # 加算済みなので生成時は1.0で適用
                pipe._combined_lora_info = {
                    'files': lora_files,
                    'weights': lora_weights,
                    'num_params': len(combined_state_dict)
                }
                
                # デバッグ: LoRAアダプターの状態確認
                if hasattr(backbone, "peft_config"):
                    print(f"✅ LoRA adapter created: {list(backbone.peft_config.keys())}")
                    print(f"✅ Active adapters: {backbone.active_adapters()}")
                else:
                    print("❌ No LoRA adapter found!")
                
            except Exception as e:
                print(f"❌ Failed to apply combined LoRA weights: {e}")
                return pipe
        else:
            print("❌ No valid LoRA weights to combine")
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
        print(f"🔧 LoRA application weight: {lora_weight}")
        
        # 複数LoRA情報の表示
        combined_info = getattr(pipe, '_combined_lora_info', None)
        if combined_info:
            print(f"📊 Combined LoRA info:")
            for i, (file, weight) in enumerate(zip(combined_info['files'], combined_info['weights'])):
                print(f"  {i+1}. {os.path.basename(file)}: weight={weight}")
            print(f"  Total parameters: {combined_info['num_params']}")
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
        
        # 各プロンプトフォルダに設定ファイルをコピー + プロンプト固有情報を追加
        main_config_path = os.path.join(save_folder, "generation_config.yaml")
        prompt_config_path = os.path.join(prompt_folder, "generation_config.yaml")
        if os.path.exists(main_config_path):
            # メイン設定を読み込み
            with open(main_config_path, 'r') as f:
                prompt_config = OmegaConf.load(f)
            
            # プロンプト固有情報を追加
            from datetime import datetime
            prompt_config.current_prompt = {
                'index': prompt_idx,
                'text': prompt,
                'folder_name': f"prompt_{prompt_idx:03d}",
                'generation_timestamp': datetime.now().isoformat()
            }
            
            # 複数LoRA情報があれば追加
            combined_info = getattr(pipe, '_combined_lora_info', None)
            if combined_info:
                prompt_config.applied_lora_info = combined_info
            
            # プロンプト固有の設定ファイルを保存
            OmegaConf.save(prompt_config, prompt_config_path)
            print(f"  📋 Generated prompt-specific generation_config.yaml")
        else:
            # メイン設定ファイルがない場合は基本的なコピー
            import shutil
            shutil.copy2(main_config_path, prompt_config_path) if os.path.exists(main_config_path) else None
        
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
        print(f"  - LoRA images (combined LoRA applied): {lora_folder}/")
        
        # 複数LoRA情報の再表示
        combined_info = getattr(pipe, '_combined_lora_info', None)
        if combined_info:
            print(f"🔗 Combined LoRA summary:")
            for i, (file, weight) in enumerate(zip(combined_info['files'], combined_info['weights'])):
                print(f"    {os.path.basename(file)}: {weight}")
    elif has_lora:
        combined_info = getattr(pipe, '_combined_lora_info', None)
        if combined_info:
            print(f"🔗 Generated with combined LoRA:")
            for i, (file, weight) in enumerate(zip(combined_info['files'], combined_info['weights'])):
                print(f"    {os.path.basename(file)}: {weight}")


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
