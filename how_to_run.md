# Dual-Process Image Generation 実行ガイド

## 0. 事前準備：Hugging Face認証

### Hugging Faceアカウントとトークンの取得

1. [Hugging Face](https://huggingface.co/)にアカウント登録・ログイン
2. [Settings > Access Tokens](https://huggingface.co/settings/tokens)でアクセストークンを作成
   - Token type: `Read` または `Write`
   - Name: 任意の名前（例：`dual-process-access`）

### 認証方法

#### 方法1: huggingface-cliを使用（推奨）
```bash
# huggingface-hubがインストールされていない場合
pip install huggingface-hub

# ログイン（トークンの入力を求められます）
huggingface-cli login
```

#### 方法2: 環境変数を設定
```bash
export HF_TOKEN=<your_hugging_face_token>
```

#### 方法3: Pythonスクリプト内で認証
```python
from huggingface_hub import login
login(token="<your_hugging_face_token>")
```

### 必要なモデルへのアクセス申請

一部のモデル（Flux等）は制限付きアクセスのため、以下の手順でアクセス申請が必要です：

1. [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)のページにアクセス
2. "Request access to this repository" ボタンをクリック
3. 利用規約に同意してアクセスを申請
4. 承認されるまで待機（通常は数分〜数時間）

### 認証確認
```bash
# 認証状態の確認
huggingface-cli whoami

# モデルへのアクセステスト
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('black-forest-labs/FLUX.1-schnell')"
```

## 1. CUDA デバイス設定の変更

### 現在の設定
- 画像生成モデル（pipe）: `cuda:1`
- ビジョン言語モデル（VLM）: `cuda:1`

### 設定変更方法

`configs/base.yaml` を編集して、使用するGPUを変更できます：

```yaml
pipe_kwargs:
  pipe_device: cuda:0  # 画像生成モデルのGPU
vlm_kwargs:
  vlm_device: cuda:0   # VLMのGPU
```

### GPU分散の例

VRAMが不足する場合、異なるGPUに分散できます：

```yaml
pipe_kwargs:
  pipe_device: cuda:0  # 1つ目のGPU
vlm_kwargs:
  vlm_device: cuda:1   # 2つ目のGPU
```

### VRAM要件
- **Flux Schnell**: 44.3GB
- **Idefics2**: 17.3GB
- **合計**: 約62GB（同一GPU使用時）

## 2. 生成プロンプトと質問の変更

### 現在のデモ設定

`configs/dataset/demo.yaml`:
```yaml
question_postfix: "Answer with Yes or No."
dataset:
  - name: man-smile
    prompt: "Photo of a man"
    qa_pairs:
      - question: "Is the man smiling?"
        answer: "Yes"
```

### 設定変更例

#### 単一の質問を変更
```yaml
question_postfix: "Answer with Yes or No."
dataset:
  - name: woman-glasses
    prompt: "Photo of a woman"
    qa_pairs:
      - question: "Is the woman wearing glasses?"
        answer: "Yes"
```

#### 複数のサンプルを設定
```yaml
question_postfix: "Answer with Yes or No."
dataset:
  - name: man-smile
    prompt: "Photo of a man"
    qa_pairs:
      - question: "Is the man smiling?"
        answer: "Yes"
  - name: cat-sleeping
    prompt: "Photo of a cat"
    qa_pairs:
      - question: "Is the cat sleeping?"
        answer: "Yes"
  - name: dog-running
    prompt: "Photo of a dog in a park"
    qa_pairs:
      - question: "Is the dog running?"
        answer: "Yes"
```

#### 複数の質問を一つのサンプルに設定
```yaml
dataset:
  - name: complex-scene
    prompt: "Photo of a woman in a park"
    qa_pairs:
      - question: "Is the woman wearing a hat?"
        answer: "Yes"
      - question: "Is she sitting on a bench?"
        answer: "Yes"
      - question: "Are there trees in the background?"
        answer: "Yes"
```

## 3. 最適化パラメータの調整

### 基本設定（`configs/base.yaml`）

```yaml
opt_kwargs:
  save_weights: True        # LoRA重みの保存
  num_opt_steps: 100       # 最適化ステップ数
  train_weight: 5          # LoRAの重み強度
  train_seed: 100          # 学習用シード

lora_kwargs:
  r: 16                    # LoRAのランク
  lora_lr: 5e-5           # LoRAの学習率
  init_lora_weights: gaussian  # 重みの初期化方法
  lora_dropout: 0.0       # ドロップアウト率

eval_kwargs:
  eval_n: 5               # 評価時の生成画像数
  eval_seed: 0            # 評価用シード
```

### パラメータ調整のガイドライン

- **`num_opt_steps`**: 最適化が収束しない場合は増加（200-500）
- **`train_weight`**: 効果が弱い場合は増加（10-20）、強すぎる場合は減少（1-3）
- **`lora_lr`**: 学習が不安定な場合は減少（1e-5）、遅い場合は増加（1e-4）
- **`r`**: より精細な調整が必要な場合は増加（32-64）

## 4. 実行方法

### バッチ処理
```bash
./run_dual_process.sh
```

### Gradio UI（インタラクティブ）
```bash
export OPENAI_API_KEY=<your_api_key>  # オプション
python3 launch_gradio.py
```

### カスタム設定での実行
```bash
pipe=schnell
vlm=idefics2
dataset=demo
save_folder=runs/custom_${dataset}_${pipe}_${vlm}

python3 script_dual_process.py \
  configs/base.yaml,configs/pipe/${pipe}.yaml,configs/vlm/${vlm}.yaml,configs/dataset/${dataset}.yaml \
  save_folder=${save_folder}
```

## 5. 結果の確認

実行結果は `runs/` ディレクトリに保存されます：

```
runs/demo_schnell_idefics2/
├── config.yaml          # 使用された設定
└── man-smile/
    └── 0/                # 最適化実行番号
        ├── weights_*.pt  # LoRA重み
        └── images/       # 生成画像
```

## 6. トラブルシューティング

### Hugging Face認証エラー
```
Cannot access gated repo for url https://huggingface.co/black-forest-labs/FLUX.1-schnell/...
Access to model black-forest-labs/FLUX.1-schnell is restricted.
```

**解決方法:**
1. `huggingface-cli login` でログイン
2. [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)でアクセス申請
3. 承認後、再実行

### VRAM不足エラー
- 異なるGPUに分散する
- `eval_kwargs.low_memory: True` を設定
- `eval_kwargs.eval_n` を減らす

### 最適化が収束しない
- `opt_kwargs.num_opt_steps` を増加
- `lora_kwargs.lora_lr` を調整
- `opt_kwargs.train_weight` を調整

### 生成品質が悪い
- プロンプトをより具体的にする
- 質問をより明確にする
- `lora_kwargs.r` を増加

## 7. 訓練済みLoRAを使った画像生成

### 基本的な使用方法（shスクリプト実行）

```bash
# 基本的なLoRA生成
./run_lora_generation.sh

# 複数LoRA組み合わせ
./run_multi_lora.sh

# 高解像度生成（1024x1024）
./run_lora_highres.sh

# 高速生成（512x512、1ステップ）
./run_lora_fast.sh
```

### shスクリプトの設定変更

各shファイルの先頭で設定を変更できます：

#### `run_lora_generation.sh` の設定例:
```bash
#!/bin/bash

# LoRA生成の設定変数
pipe=sana              # 使用するパイプライン（sana/schnell/dev）
generation_config=lora_demo  # 生成設定ファイル名
experiment_name=demo_sana_idefics2  # 実験フォルダ名

# 出力フォルダ
save_folder=lora_generated/${generation_config}_${pipe}
```

### 設定ファイルの作成

#### `configs/generation/my_lora.yaml` の例:
```yaml
# 単一LoRAファイルの使用
lora_generation:
  lora_files:
    - "runs/demo_sana_idefics2/man-smile/0/lora_o-000099.pt"
  lora_weights:
    - 1.0

generation:
  prompts:
    - "Photo of a man"
    - "Portrait of a man in sunlight"
  num_images_per_prompt: 4
  seed: 42

save_folder: "my_lora_output"
```

#### 複数LoRAの組み合わせ例:
```yaml
# 複数LoRAファイルの組み合わせ
lora_generation:
  lora_files:
    - "runs/demo_sana_idefics2/man-smile/0/lora_o-000099.pt"
    - "runs/other_experiment/woman-glasses/0/lora_o-000050.pt"
  lora_weights:
    - 0.8  # 最初のLoRAの重み
    - 0.6  # 2番目のLoRAの重み

generation:
  prompts:
    - "Photo of a man smiling and wearing glasses"
  num_images_per_prompt: 6
  seed: 123
```

### 用意されているshスクリプト

1. **`run_lora_generation.sh`** - 基本的なLoRA生成
2. **`run_multi_lora.sh`** - 複数LoRA組み合わせ
3. **`run_lora_highres.sh`** - 高解像度生成（1024x1024）
4. **`run_lora_fast.sh`** - 高速生成（512x512、1ステップ）

### 直接Pythonコマンドでの実行（上級者向け）

```bash
# 基本実行
python3 generate_with_lora.py \
  configs/base.yaml,configs/pipe/sana.yaml,configs/generation/lora_demo.yaml

# カスタム出力フォルダ
python3 generate_with_lora.py \
  configs/base.yaml,configs/pipe/sana.yaml,configs/generation/lora_demo.yaml \
  save_folder=my_custom_output

# デバイス変更
python3 generate_with_lora.py \
  configs/base.yaml,configs/pipe/sana.yaml,configs/generation/lora_demo.yaml \
  pipe_kwargs.pipe_device=cuda:0
```

### 生成結果の構造

```
generated_with_lora/
├── generation_config.yaml    # 使用された設定
├── prompt_000/               # 1つ目のプロンプト
│   ├── prompt.txt           # プロンプトテキスト
│   ├── image_000_seed_42.png
│   ├── image_001_seed_43.png
│   └── ...
├── prompt_001/               # 2つ目のプロンプト
│   └── ...
└── ...
```

### LoRA重みの調整

- `lora_weights: [1.0]`: フル効果
- `lora_weights: [0.5]`: 50%効果
- `lora_weights: [0.0]`: 効果なし（ベースモデル）
- `lora_weights: [1.2]`: 120%効果（過強調）
