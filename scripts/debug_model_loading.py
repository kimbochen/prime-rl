"""Debug script to reproduce model loading and conversion logic without launching a full training run.

Usage:
    uv run python scripts/debug_model_loading.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --impl custom --ep 8
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, PreTrainedModelPrimeRL, supports_custom_impl


def load_state_dict_keys(save_dir: Path) -> list[str]:
    keys: list[str] = []
    for safetensor_path in save_dir.glob("*.safetensors"):
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--impl", choices=["custom", "hf", "auto"], default="auto")
    parser.add_argument("--ep", type=int, default=0, help="Expert parallelism degree")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    print(f"=== Loading config for {args.model} ===")
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print(f"model_type: {model_config.model_type}")
    print(f"tie_word_embeddings: {model_config.tie_word_embeddings}")
    print(f"supports_custom_impl: {supports_custom_impl(model_config)}")

    # Determine impl
    impl = args.impl
    if impl == "auto":
        impl = "custom" if supports_custom_impl(model_config) else "hf"
    print(f"Using impl: {impl}")

    # Load model to meta device (same as trainer: from_config for meta)
    print(f"\n=== Loading model to meta device ===")
    if impl == "custom":
        model = AutoModelForCausalLMPrimeRL.from_config(model_config, torch_dtype=torch.bfloat16)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(
            model_config, torch_dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code,
        )

    is_prime = isinstance(model, PreTrainedModelPrimeRL)
    print(f"isinstance(PreTrainedModelPrimeRL): {is_prime}")
    print(f"model class: {type(model).__name__}")

    # Get model state dict keys
    print(f"\n=== Model state dict keys ===")
    model_keys = list(model.state_dict().keys())
    print(f"Total model keys: {len(model_keys)}")
    print(f"First 10 keys: {model_keys[:10]}")

    expert_keys = [k for k in model_keys if "experts" in k]
    print(f"\nExpert keys ({len(expert_keys)} total), first 10:")
    for k in expert_keys[:10]:
        print(f"  {k}")

    lm_head_keys = [k for k in model_keys if "lm_head" in k]
    print(f"\nlm_head keys: {lm_head_keys}")

    embed_keys = [k for k in model_keys if "embed_tokens" in k]
    print(f"embed_tokens keys: {embed_keys}")

    # Check format detection
    if is_prime:
        model_keys_dict = dict.fromkeys(model_keys)
        is_prime_format = model.is_prime_state_dict(model_keys_dict)
        is_hf_format = model.is_hf_state_dict(model_keys_dict)
        print(f"\nModel state dict format detection:")
        print(f"  is_prime_state_dict: {is_prime_format}")
        print(f"  is_hf_state_dict: {is_hf_format}")

    # Load snapshot and check its format
    print(f"\n=== Snapshot keys ===")
    if Path(args.model).exists():
        snapshot_path = Path(args.model)
    else:
        snapshot_path = Path(snapshot_download(repo_id=args.model, repo_type="model"))
    print(f"Snapshot path: {snapshot_path}")

    snapshot_keys = load_state_dict_keys(snapshot_path)
    print(f"Total snapshot keys: {len(snapshot_keys)}")

    snapshot_expert_keys = [k for k in snapshot_keys if "experts" in k]
    print(f"Expert keys ({len(snapshot_expert_keys)} total), first 10:")
    for k in snapshot_expert_keys[:10]:
        print(f"  {k}")

    snapshot_lm_head = [k for k in snapshot_keys if "lm_head" in k]
    print(f"\nlm_head keys: {snapshot_lm_head}")

    if is_prime:
        snapshot_keys_dict = dict.fromkeys(snapshot_keys)
        snap_is_prime = model.is_prime_state_dict(snapshot_keys_dict)
        snap_is_hf = model.is_hf_state_dict(snapshot_keys_dict)
        print(f"\nSnapshot format detection:")
        print(f"  is_prime_state_dict: {snap_is_prime}")
        print(f"  is_hf_state_dict: {snap_is_hf}")

        # Determine what would happen
        print(f"\n=== Conversion decision ===")
        if snap_is_hf and is_prime_format:
            print("WOULD CONVERT: HF snapshot -> PrimeRL format (save to snapshot_path/prime)")
        elif snap_is_prime and is_hf_format:
            print("WOULD CONVERT: PrimeRL snapshot -> HF format (save to snapshot_path/hf)")
        else:
            print("NO CONVERSION — loading snapshot as-is")
            print(f"  snap_is_hf={snap_is_hf}, snap_is_prime={snap_is_prime}")
            print(f"  model_is_hf={is_hf_format}, model_is_prime={is_prime_format}")

            # Check for key mismatches
            model_set = set(model_keys)
            snap_set = set(snapshot_keys)
            missing_in_snap = model_set - snap_set
            missing_in_model = snap_set - model_set
            if missing_in_snap:
                print(f"\n  Keys in model but NOT in snapshot ({len(missing_in_snap)}):")
                for k in sorted(missing_in_snap)[:20]:
                    print(f"    {k}")
            if missing_in_model:
                print(f"\n  Keys in snapshot but NOT in model ({len(missing_in_model)}):")
                for k in sorted(missing_in_model)[:20]:
                    print(f"    {k}")


if __name__ == "__main__":
    main()
