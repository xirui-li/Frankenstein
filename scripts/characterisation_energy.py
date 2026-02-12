"""
Frobenius Norm Analyzer - Simplified Version
Computes Frobenius norm of weight deltas for SFT and RL phases.

Usage:
  python frobenius_norm_analyzer.py \
    --base Qwen/Qwen2.5-VL-7B-Instruct \
    --sft Video-R1/Qwen2.5-VL-7B-COT-SFT \
    --rl Video-R1/Video-R1-7B \
    --output results/fro_norm.json

  # For LLM (not VLM)
  python frobenius_norm_analyzer.py --llm \
    --base Qwen/Qwen2.5-7B \
    --sft Qwen/Qwen2.5-7B-Instruct \
    --rl some-rl-model \
    --output results/fro_norm_llm.json
"""

import torch
import numpy as np
import json
import argparse
import gc
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM


def get_layer_info(name: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Unified Layer Indexing: Vision < 0, Projector = 0, LLM > 0
    """
    # 1. Vision Encoder (-32 to -1)
    match_vis = re.search(r'visual\.(?:blocks|layers)\.(\d+)\.', name)
    if match_vis:
        return -32 + int(match_vis.group(1)), "Vision Encoder"

    # 2. Projector (0)
    if any(x in name for x in ['merger', 'mm_projector', 'visual.projector', 'inc_projector']):
        return 0, "Projector"

    # 3. LLM (1 to 32+)
    match_llm = re.search(r'(?:^|\.|model\.)layers\.(\d+)\.', name)
    if match_llm:
        return int(match_llm.group(1)) + 1, "LLM Backbone"

    return None, None


def get_component_type(name: str) -> str:
    """Classify parameter as Attention, MLP, or Other."""
    if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn', 'qkv']):
        return 'Attention'
    elif any(x in name for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
        return 'MLP'
    else:
        return 'Other'


def save_json(data: Any, path: Path):
    """Save data to JSON with numpy conversion."""
    def default_converter(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=default_converter)


class FrobeniusNormAnalyzer:
    """
    Computes Frobenius norm of weight deltas for SFT and RL phases.
    - SFT delta: SFT - Base
    - RL delta: RL - SFT
    """

    def __init__(self, base_path: str, sft_path: str, rl_path: str, is_llm: bool = False):
        self.base_path = base_path
        self.sft_path = sft_path
        self.rl_path = rl_path
        self.is_llm = is_llm

    def _load_model(self, path: str):
        print(f"Loading {'LLM' if self.is_llm else 'VLM'}: {path}")
        if self.is_llm:
            model = AutoModelForCausalLM.from_pretrained(
                path, device_map="cpu", torch_dtype=torch.float16
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, device_map="cpu", torch_dtype=torch.float16
            )
        return model

    def run(self) -> Dict[str, Any]:
        # Phase 1: Load Base and SFT
        print("\n" + "=" * 50)
        print("Phase 1: Loading Base & SFT models...")
        print("=" * 50)

        model_base = self._load_model(self.base_path)
        sd_base = model_base.state_dict()
        del model_base
        gc.collect()

        model_sft = self._load_model(self.sft_path)
        sd_sft = model_sft.state_dict()
        del model_sft
        gc.collect()

        # Identify target keys
        target_keys = []
        for k in sd_sft.keys():
            if k in sd_base and 'weight' in k and 'norm' not in k.lower():
                idx, _ = get_layer_info(k)
                if idx is not None:
                    target_keys.append(k)

        print(f"Target parameters: {len(target_keys)}")

        # Compute SFT deltas and store
        print("\nComputing SFT deltas (Base -> SFT)...")
        delta_sft_store = {}
        for k in tqdm(target_keys):
            delta_sft_store[k] = sd_sft[k] - sd_base[k]

        del sd_base
        gc.collect()

        # Phase 2: Load RL
        print("\n" + "=" * 50)
        print("Phase 2: Loading RL model...")
        print("=" * 50)

        model_rl = self._load_model(self.rl_path)
        sd_rl = model_rl.state_dict()
        del model_rl
        gc.collect()

        # Phase 3: Compute Frobenius norms
        print("\n" + "=" * 50)
        print("Phase 3: Computing Frobenius norms...")
        print("=" * 50)

        results = {
            'metadata': {
                'base_model': self.base_path,
                'sft_model': self.sft_path,
                'rl_model': self.rl_path,
            },
            'layers': [],
            'summary': [],
        }

        layer_data = defaultdict(lambda: {'sft': [], 'rl': []})

        for k in tqdm(target_keys):
            if k not in sd_rl or k not in delta_sft_store:
                continue

            delta_sft = delta_sft_store[k].float()
            delta_rl = (sd_rl[k] - sd_sft[k]).float()

            # Compute Frobenius norms
            fro_sft = torch.norm(delta_sft).item()
            fro_rl = torch.norm(delta_rl).item()

            # Original weight norm (from SFT, as reference)
            w_norm = torch.norm(sd_sft[k].float()).item() + 1e-8

            idx, section = get_layer_info(k)
            component = get_component_type(k)

            entry = {
                'param_name': k,
                'layer_idx': idx,
                'section': section,
                'component': component,
                'sft': {
                    'fro_norm': fro_sft,
                    'relative_norm': fro_sft / w_norm,
                },
                'rl': {
                    'fro_norm': fro_rl,
                    'relative_norm': fro_rl / w_norm,
                },
            }

            results['layers'].append(entry)
            layer_data[idx]['sft'].append(entry['sft'])
            layer_data[idx]['rl'].append(entry['rl'])

        # Aggregate by layer
        print("\nComputing layer-wise summary...")
        for idx in sorted(layer_data.keys()):
            sft_list = layer_data[idx]['sft']
            rl_list = layer_data[idx]['rl']

            if not sft_list or not rl_list:
                continue

            # Get section info
            section = None
            for entry in results['layers']:
                if entry['layer_idx'] == idx:
                    section = entry['section']
                    break

            results['summary'].append({
                'layer_idx': idx,
                'section': section,
                'n_params': len(sft_list),
                'sft': {
                    'fro_norm_mean': np.mean([e['fro_norm'] for e in sft_list]),
                    'fro_norm_sum': sum(e['fro_norm'] for e in sft_list),
                    'relative_norm_mean': np.mean([e['relative_norm'] for e in sft_list]),
                },
                'rl': {
                    'fro_norm_mean': np.mean([e['fro_norm'] for e in rl_list]),
                    'fro_norm_sum': sum(e['fro_norm'] for e in rl_list),
                    'relative_norm_mean': np.mean([e['relative_norm'] for e in rl_list]),
                },
            })

        # Print summary by section
        print("\n" + "=" * 50)
        print("Summary by Section:")
        print("=" * 50)

        section_stats = defaultdict(lambda: {'sft': [], 'rl': []})
        for entry in results['layers']:
            section_stats[entry['section']]['sft'].append(entry['sft']['fro_norm'])
            section_stats[entry['section']]['rl'].append(entry['rl']['fro_norm'])

        for section in ['Vision Encoder', 'Projector', 'LLM Backbone']:
            if section in section_stats:
                sft_mean = np.mean(section_stats[section]['sft'])
                rl_mean = np.mean(section_stats[section]['rl'])
                print(f"  {section:15s}: SFT fro_norm={sft_mean:.4f}, RL fro_norm={rl_mean:.4f}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Frobenius Norm Analyzer (Base -> SFT -> RL)"
    )
    parser.add_argument('--base', required=True, help="Base model path")
    parser.add_argument('--sft', required=True, help="SFT model path")
    parser.add_argument('--rl', required=True, help="RL model path")
    parser.add_argument('--output', default='fro_norm.json', help="Output JSON path")
    parser.add_argument('--llm', action='store_true', help="Use AutoModelForCausalLM (LLM mode)")

    args = parser.parse_args()

    analyzer = FrobeniusNormAnalyzer(
        base_path=args.base,
        sft_path=args.sft,
        rl_path=args.rl,
        is_llm=args.llm,
    )
    results = analyzer.run()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_path)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
