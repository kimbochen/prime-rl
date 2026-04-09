"""Convert HF checkpoint to PrimeRL format for Qwen3 MoE models.

Usage:
    uv run python scripts/convert_hf_to_prime.py \
        --input /mnt/vast/kb_cache/hf_hub_cache/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d \
        --output /mnt/vast/kb_cache/hf_hub_cache/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d/prime
"""

import argparse
from pathlib import Path

from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import convert_hf_to_tt_moe
from prime_rl.trainer.weights import load_state_dict, load_state_dict_keys, save_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to PrimeRL format")
    parser.add_argument("--input", required=True, help="Path to HF snapshot dir")
    parser.add_argument("--output", required=True, help="Path to save converted checkpoint")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists():
        print(f"Output path {output_path} already exists, skipping")
        return

    keys = load_state_dict_keys(input_path)
    print(f"Loading {len(keys)} keys from {input_path}")

    state_dict = load_state_dict(input_path)
    print(f"Loaded state dict, converting HF -> PrimeRL format...")

    convert_hf_to_tt_moe(state_dict)
    print(f"Converted. Saving to {output_path}...")

    save_state_dict(state_dict, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
