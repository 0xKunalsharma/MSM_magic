# ============================================================
# 1. SETUP & AUTHENTICATION
# ============================================================
!pip -q install huggingface_hub safetensors

from huggingface_hub import hf_hub_download, login
import json
import os
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from collections import defaultdict
from huggingface_hub import HfApi, upload_folder

login()

# ============================================================
# 2. INSPECT LORA ADAPTER & BASE MODEL
# ============================================================
repo = "chloeli/qwen-3-14b-rules-aug-spec-msm"
base = "Qwen/Qwen3-14B"

# Read adapter config
path = hf_hub_download(repo, "adapter_config.json")
with open(path) as f:
    config = json.load(f)

print("Adapter Config:")
for key in ["base_model_name_or_path", "peft_type", "r", "lora_alpha", "target_modules"]:
    print(f"  {key}: {config.get(key)}")

# Inspect adapter tensors
adapter_path = hf_hub_download(repo, "adapter_model.safetensors")
with safe_open(adapter_path, framework="np") as f:
    keys = list(f.keys())
    print(f"\nNumber of adapter tensors: {len(keys)}")
    for key in keys[:3]:
        print(f"  {key} {f.get_tensor(key).shape} {f.get_tensor(key).dtype}")

# Verify LoRA A/B pairs
pairs = defaultdict(set)
for key in keys:
    if ".lora_A.weight" in key:
        pairs[key.replace(".lora_A.weight", "")].add("A")
    elif ".lora_B.weight" in key:
        pairs[key.replace(".lora_B.weight", "")].add("B")

incomplete = {name: sides for name, sides in pairs.items() if sides != {"A", "B"}}
print(f"\nLoRA modules: {len(pairs)}")
print(f"Incomplete pairs: {incomplete}")

# ============================================================
# 3. MERGE LORA INTO BASE MODEL
# ============================================================
OUT = "/content/qwen3-14b-rules-aug-spec-msm-merged"
os.makedirs(OUT, exist_ok=True)

# Load base index and adapter
index_path = hf_hub_download(base, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)
weight_map = index["weight_map"]
shards = list(dict.fromkeys(weight_map.values()))

adapter = load_file(adapter_path, device="cpu")
scale = config["lora_alpha"] / config["r"]

# Map adapter keys to base keys
targets = {}
for prefix, pair in pairs.items():
    target = prefix[len("base_model.model."):] + ".weight"
    if target not in weight_map:
        raise KeyError(f"No matching base tensor: {target}")
    targets[target] = pair

# Merge shard by shard
for i, shard in enumerate(shards, 1):
    print(f"[{i}/{len(shards)}] Processing {shard}", flush=True)
    src = hf_hub_download(base, shard)
    weights = load_file(src, device="cpu")

    for target, pair in targets.items():
        if weight_map[target] != shard:
            continue
        
        W = weights[target]
        A = adapter[pair["A"]]
        B = adapter[pair["B"]]

        # Compute delta in FP32 and add to base weight
        delta = (B.float() @ A.float()) * scale
        weights[target] = (W.float() + delta).to(W.dtype)

    save_file(weights, os.path.join(OUT, shard))
    del weights

# Copy metadata
for name in ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"]:
    try:
        src = hf_hub_download(base, name)
        shutil.copy2(src, os.path.join(OUT, name))
    except Exception:
        pass

# ============================================================
# 4. BUILD LLAMA.CPP & CONVERT TO GGUF
# ============================================================
!git clone --depth 1 https://github.com/ggml-org/llama.cpp
!pip -q install -r llama.cpp/requirements.txt
!cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF
!cmake --build llama.cpp/build --config Release -j 4

# Convert to F16 GGUF
!python llama.cpp/convert_hf_to_gguf.py \
  /content/qwen3-14b-rules-aug-spec-msm-merged \
  --outfile /content/qwen3-14b-msm-f16.gguf \
  --outtype f16

# ============================================================
# 5. QUANTIZE & UPLOAD
# ============================================================
# Quantize to Q4_K_M
!llama.cpp/build/bin/llama-quantize \
  /content/qwen3-14b-msm-f16.gguf \
  /content/qwen3-14b-msm-Q4_K_M.gguf \
  Q4_K_M

# Upload to Hugging Face
upload_dir = "/content/upload_model"
os.makedirs(upload_dir, exist_ok=True)

gguf_path = "/content/qwen3-14b-msm-Q4_K_M.gguf"
shutil.copy(gguf_path, upload_dir)

yaml_config = """model_type: qwen3
quantization: Q4_K_M
base_model: Qwen/Qwen3-14B
"""
with open(os.path.join(upload_dir, "config.yaml"), "w") as f:
    f.write(yaml_config)

upload_folder(
    folder_path=upload_dir,
    repo_id="neo-kunal/qwen3-14b-msm-Q4_K_M",
    repo_type="model"
)
print("Upload complete!")
