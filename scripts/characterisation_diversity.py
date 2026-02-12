"""
Layer-wise SVD Analyzer for SFT vs RL Comparison

Computes SVD metrics on weight deltas for ALL layers (not sampled),
enabling comprehensive spectral comparison between SFT and RL training phases.

Usage:
  # Compute layer-wise SVD for SFT and RL deltas
  python layerwise_svd_analyzer.py compute \
    --base Qwen/Qwen2.5-VL-7B-Instruct \
    --sft OpenMMReasoner/OpenMMReasoner-ColdStart \
    --rl OpenMMReasoner/OpenMMReasoner-RL \
    --output results/layerwise_svd.json

  # Generate comparison plots from computed data
  python layerwise_svd_analyzer.py plot \
    --input results/layerwise_svd.json \
    --output-dir results/svd_plots
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
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM


# ==========================================
# 1. UTILITIES
# ==========================================

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


def get_llm_region(idx: int, num_layers: int) -> str:
    """Classify LLM layer as early/middle/late (1/3 each)."""
    if idx <= 0:
        return None
    third = num_layers // 3
    layer_idx = idx - 1  # Convert back to 0-indexed
    if layer_idx < third:
        return "early"
    elif layer_idx < 2 * third:
        return "middle"
    else:
        return "late"


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


def load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


# ==========================================
# 2. SVD METRICS COMPUTATION
# ==========================================

def compute_svd_metrics(delta: torch.Tensor, topk: int = 50) -> Optional[Dict[str, Any]]:
    """
    Compute comprehensive SVD metrics on a weight delta matrix.

    Returns:
        Dict with spectral_norm, stable_rank, effective_rank,
        nuclear_norm, fro_norm, condition_number, top_k singular values
    """
    if delta.dim() < 2:
        return None

    try:
        delta = delta.float()
        if delta.numel() > 100_000_000:  # Safety limit for very large matrices
            return None

        if delta.dim() > 2:
            delta = delta.flatten(1)

        # Compute singular values
        S = torch.linalg.svdvals(delta)

        # Basic norms
        spectral_norm = S[0].item()
        nuclear_norm = S.sum().item()
        fro_norm = torch.linalg.norm(delta).item()

        # Stable rank: ||W||_F^2 / ||W||_2^2
        stable_rank = (fro_norm ** 2) / (spectral_norm ** 2 + 1e-12)

        # Effective rank (exponential of entropy)
        S_pos = S[S > 1e-12]
        if len(S_pos) > 0:
            p = S_pos / S_pos.sum()
            entropy = -(p * torch.log(p)).sum().item()
            effective_rank = float(np.exp(entropy))
        else:
            effective_rank = 0.0

        # Condition number
        condition_number = (S[0] / (S[-1] + 1e-12)).item() if len(S) > 1 else 1.0

        # Log spectral volume
        log_volume = torch.log(S[:topk] + 1e-12).sum().item()

        return {
            'spectral_norm': spectral_norm,
            'nuclear_norm': nuclear_norm,
            'fro_norm': fro_norm,
            'stable_rank': stable_rank,
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'log_volume': log_volume,
            'singular_values': S[:topk].cpu().numpy().tolist(),
        }
    except Exception as e:
        print(f"  [WARN] SVD failed: {e}")
        return None


# ==========================================
# 3. LAYER-WISE SVD ANALYZER
# ==========================================

class LayerwiseSVDAnalyzer:
    """
    Computes SVD metrics on weight deltas for ALL layers,
    comparing SFT (Base->SFT) and RL (SFT->RL) phases.
    """

    def __init__(self, base_path: str, sft_path: str, rl_path: str,
                 is_llm: bool = False, topk: int = 50,
                 weight_filter: str = "down_proj"):
        self.base_path = base_path
        self.sft_path = sft_path
        self.rl_path = rl_path
        self.is_llm = is_llm
        self.topk = topk
        self.weight_filter = weight_filter  # Focus on specific weight type

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

    def _should_analyze(self, name: str) -> bool:
        """Filter which parameters to analyze."""
        # Must be a weight matrix (not bias, not norm)
        if "weight" not in name:
            return False
        if "norm" in name.lower():
            return False

        # Must have valid layer index
        idx, section = get_layer_info(name)
        if idx is None:
            return False

        # Filter by weight type if specified
        if self.weight_filter:
            if self.weight_filter not in name:
                return False

        return True

    def run(self) -> Dict[str, Any]:
        """
        Main computation: load models, compute deltas, run SVD on all layers.
        """
        # Phase 1: Load Base and SFT, compute delta_sft
        print("\n" + "=" * 60)
        print("Phase 1: Loading Base & SFT models...")
        print("=" * 60)

        model_base = self._load_model(self.base_path)
        sd_base = model_base.state_dict()
        del model_base
        gc.collect()

        model_sft = self._load_model(self.sft_path)
        sd_sft = model_sft.state_dict()
        del model_sft
        gc.collect()

        # Identify target keys
        target_keys = [k for k in sd_sft.keys() if self._should_analyze(k)]
        print(f"\nTarget parameters: {len(target_keys)}")

        # Compute delta_sft and store
        print("\nComputing SFT deltas (Base -> SFT)...")
        delta_sft_store = {}
        for k in tqdm(target_keys):
            if k in sd_base:
                delta_sft_store[k] = sd_sft[k] - sd_base[k]

        del sd_base
        gc.collect()

        # Phase 2: Load RL, compute delta_rl
        print("\n" + "=" * 60)
        print("Phase 2: Loading RL model...")
        print("=" * 60)

        model_rl = self._load_model(self.rl_path)
        sd_rl = model_rl.state_dict()
        del model_rl
        gc.collect()

        # Phase 3: Compute SVD for all layers
        print("\n" + "=" * 60)
        print("Phase 3: Computing SVD metrics for all layers...")
        print("=" * 60)

        # Determine number of LLM layers for region classification
        llm_indices = []
        for k in target_keys:
            idx, section = get_layer_info(k)
            if section == "LLM Backbone":
                llm_indices.append(idx)
        num_llm_layers = max(llm_indices) if llm_indices else 28

        results = {
            'metadata': {
                'base_model': self.base_path,
                'sft_model': self.sft_path,
                'rl_model': self.rl_path,
                'weight_filter': self.weight_filter,
                'topk': self.topk,
                'num_llm_layers': num_llm_layers,
            },
            'layers': []
        }

        # Group by layer index for aggregation
        layer_data = defaultdict(lambda: {'sft': [], 'rl': []})

        for k in tqdm(target_keys):
            if k not in sd_rl or k not in delta_sft_store:
                continue

            # Compute delta_rl
            delta_rl = sd_rl[k] - sd_sft[k]
            delta_sft = delta_sft_store[k]

            # Compute SVD metrics
            sft_metrics = compute_svd_metrics(delta_sft, topk=self.topk)
            rl_metrics = compute_svd_metrics(delta_rl, topk=self.topk)

            if sft_metrics is None or rl_metrics is None:
                continue

            idx, section = get_layer_info(k)
            component = get_component_type(k)
            region = get_llm_region(idx, num_llm_layers) if section == "LLM Backbone" else None

            layer_entry = {
                'param_name': k,
                'layer_idx': idx,
                'section': section,
                'component': component,
                'region': region,
                'sft': sft_metrics,
                'rl': rl_metrics,
            }

            results['layers'].append(layer_entry)
            layer_data[idx]['sft'].append(sft_metrics)
            layer_data[idx]['rl'].append(rl_metrics)

        # Compute per-layer aggregated summary
        print("\nComputing layer-wise summary...")
        summary = []
        for idx in sorted(layer_data.keys()):
            sft_list = layer_data[idx]['sft']
            rl_list = layer_data[idx]['rl']

            if not sft_list or not rl_list:
                continue

            # Get section info from first entry
            section = None
            for entry in results['layers']:
                if entry['layer_idx'] == idx:
                    section = entry['section']
                    break

            region = get_llm_region(idx, num_llm_layers) if section == "LLM Backbone" else None

            summary.append({
                'layer_idx': idx,
                'section': section,
                'region': region,
                'n_params': len(sft_list),
                'sft': {
                    'spectral_norm_mean': np.mean([m['spectral_norm'] for m in sft_list]),
                    'stable_rank_mean': np.mean([m['stable_rank'] for m in sft_list]),
                    'effective_rank_mean': np.mean([m['effective_rank'] for m in sft_list]),
                    'log_volume_mean': np.mean([m['log_volume'] for m in sft_list]),
                },
                'rl': {
                    'spectral_norm_mean': np.mean([m['spectral_norm'] for m in rl_list]),
                    'stable_rank_mean': np.mean([m['stable_rank'] for m in rl_list]),
                    'effective_rank_mean': np.mean([m['effective_rank'] for m in rl_list]),
                    'log_volume_mean': np.mean([m['log_volume'] for m in rl_list]),
                },
            })

        results['summary'] = summary

        # Print summary
        print("\n" + "=" * 60)
        print("Summary by Section:")
        print("=" * 60)
        section_stats = defaultdict(lambda: {'sft': [], 'rl': []})
        for entry in results['layers']:
            section_stats[entry['section']]['sft'].append(entry['sft']['stable_rank'])
            section_stats[entry['section']]['rl'].append(entry['rl']['stable_rank'])

        for section in ['Vision Encoder', 'Projector', 'LLM Backbone']:
            if section in section_stats:
                sft_mean = np.mean(section_stats[section]['sft'])
                rl_mean = np.mean(section_stats[section]['rl'])
                print(f"  {section:15s}: SFT stable_rank={sft_mean:.2f}, RL stable_rank={rl_mean:.2f}")

        return results


# ==========================================
# 4. VISUALIZATION
# ==========================================

class LayerwiseSVDVisualizer:
    """Generate plots from layer-wise SVD analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _add_region_background(self, ax=None):
        """Add shaded regions for Vision/Projector/LLM."""
        if ax is None:
            ax = plt.gca()
        ax.axvspan(-35, -0.5, color='#c084fc', alpha=0.08, label='Vision')
        ax.axvspan(-0.5, 0.5, color='#f87171', alpha=0.12, label='Projector')
        ax.axvspan(0.5, 35, color='#4ade80', alpha=0.08, label='LLM')

    def plot_all(self, data: Dict[str, Any]):
        """Generate all plots from layer-wise SVD data."""
        print("\nGenerating plots...")

        self.plot_spectral_norm_comparison(data)
        self.plot_stable_rank_comparison(data)
        self.plot_effective_rank_comparison(data)
        self.plot_svd_spectrum_comparison(data)
        self.plot_log_volume_comparison(data)

        print(f"\nAll plots saved to: {self.output_dir}")

    def plot_spectral_norm_comparison(self, data: Dict[str, Any]):
        """Spectral norm (max singular value) across layers."""
        summary = data['summary']

        df = pd.DataFrame([
            {'layer_idx': s['layer_idx'], 'value': s['sft']['spectral_norm_mean'], 'phase': 'SFT'}
            for s in summary
        ] + [
            {'layer_idx': s['layer_idx'], 'value': s['rl']['spectral_norm_mean'], 'phase': 'RL'}
            for s in summary
        ])

        plt.figure(figsize=(14, 6))
        self._add_region_background()

        sns.lineplot(data=df, x='layer_idx', y='value', hue='phase',
                     style='phase', markers=True, dashes=False,
                     palette={'SFT': '#d62728', 'RL': '#1f77b4'}, linewidth=2)

        plt.title("Spectral Norm (Max Singular Value) - SFT vs RL", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Spectral Norm", fontsize=12)
        plt.legend(title="Phase")
        plt.tight_layout()
        plt.savefig(self.output_dir / "spectral_norm_comparison.png", dpi=150)
        plt.close()
        print("  Saved: spectral_norm_comparison.png")

    def plot_stable_rank_comparison(self, data: Dict[str, Any]):
        """Stable rank across layers."""
        summary = data['summary']

        df = pd.DataFrame([
            {'layer_idx': s['layer_idx'], 'value': s['sft']['stable_rank_mean'], 'phase': 'SFT'}
            for s in summary
        ] + [
            {'layer_idx': s['layer_idx'], 'value': s['rl']['stable_rank_mean'], 'phase': 'RL'}
            for s in summary
        ])

        plt.figure(figsize=(14, 6))
        self._add_region_background()

        sns.lineplot(data=df, x='layer_idx', y='value', hue='phase',
                     style='phase', markers=True, dashes=False,
                     palette={'SFT': '#d62728', 'RL': '#1f77b4'}, linewidth=2)

        plt.title("Stable Rank - SFT vs RL", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Stable Rank", fontsize=12)
        plt.legend(title="Phase")
        plt.tight_layout()
        plt.savefig(self.output_dir / "stable_rank_comparison.png", dpi=150)
        plt.close()
        print("  Saved: stable_rank_comparison.png")

    def plot_effective_rank_comparison(self, data: Dict[str, Any]):
        """Effective rank across layers."""
        summary = data['summary']

        df = pd.DataFrame([
            {'layer_idx': s['layer_idx'], 'value': s['sft']['effective_rank_mean'], 'phase': 'SFT'}
            for s in summary
        ] + [
            {'layer_idx': s['layer_idx'], 'value': s['rl']['effective_rank_mean'], 'phase': 'RL'}
            for s in summary
        ])

        plt.figure(figsize=(14, 6))
        self._add_region_background()

        sns.lineplot(data=df, x='layer_idx', y='value', hue='phase',
                     style='phase', markers=True, dashes=False,
                     palette={'SFT': '#d62728', 'RL': '#1f77b4'}, linewidth=2)

        plt.title("Effective Rank - SFT vs RL", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Effective Rank", fontsize=12)
        plt.legend(title="Phase")
        plt.tight_layout()
        plt.savefig(self.output_dir / "effective_rank_comparison.png", dpi=150)
        plt.close()
        print("  Saved: effective_rank_comparison.png")

    def plot_log_volume_comparison(self, data: Dict[str, Any]):
        """Log spectral volume across layers."""
        summary = data['summary']

        df = pd.DataFrame([
            {'layer_idx': s['layer_idx'], 'value': s['sft']['log_volume_mean'], 'phase': 'SFT'}
            for s in summary
        ] + [
            {'layer_idx': s['layer_idx'], 'value': s['rl']['log_volume_mean'], 'phase': 'RL'}
            for s in summary
        ])

        plt.figure(figsize=(14, 6))
        self._add_region_background()

        sns.lineplot(data=df, x='layer_idx', y='value', hue='phase',
                     style='phase', markers=True, dashes=False,
                     palette={'SFT': '#d62728', 'RL': '#1f77b4'}, linewidth=2)

        plt.title("Log Spectral Volume - SFT vs RL", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel(r"Log Volume ($\sum \log \sigma_i$)", fontsize=12)
        plt.legend(title="Phase")
        plt.tight_layout()
        plt.savefig(self.output_dir / "log_volume_comparison.png", dpi=150)
        plt.close()
        print("  Saved: log_volume_comparison.png")

    def plot_svd_spectrum_comparison(self, data: Dict[str, Any]):
        """
        Plot singular value spectrum for Early/Mid/Late LLM layers.
        Similar to the original svd_comparison.png but with all-layer data.
        """
        layers = data['layers']
        num_llm_layers = data['metadata'].get('num_llm_layers', 28)

        # Group by region
        early_layers = [l for l in layers if l.get('region') == 'early']
        mid_layers = [l for l in layers if l.get('region') == 'middle']
        late_layers = [l for l in layers if l.get('region') == 'late']

        def get_representative(layer_list):
            """Get the layer closest to the middle of the list."""
            if not layer_list:
                return None
            sorted_layers = sorted(layer_list, key=lambda x: x['layer_idx'])
            return sorted_layers[len(sorted_layers) // 2]

        regions = [
            ('Early', get_representative(early_layers), '#2ca02c'),
            ('Mid', get_representative(mid_layers), '#ff7f0e'),
            ('Late', get_representative(late_layers), '#1f77b4'),
        ]

        plt.figure(figsize=(12, 7))

        for region_name, layer, color in regions:
            if layer is None:
                continue

            sft_sv = np.array(layer['sft']['singular_values'])
            rl_sv = np.array(layer['rl']['singular_values'])

            if len(sft_sv) > 0 and sft_sv[0] > 0:
                plt.plot(sft_sv / sft_sv[0], color=color, linestyle='-',
                        alpha=0.5, linewidth=2, label=f'{region_name} (SFT)')
            if len(rl_sv) > 0 and rl_sv[0] > 0:
                plt.plot(rl_sv / rl_sv[0], color=color, linestyle='--',
                        linewidth=2.5, label=f'{region_name} (RL)')

        plt.yscale('log')
        plt.xlabel("Singular Value Index", fontsize=12)
        plt.ylabel("Normalized Singular Value (log scale)", fontsize=12)
        plt.title("SVD Spectrum: SFT (solid) vs RL (dashed)\nEarly/Mid/Late LLM Layers", fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "svd_spectrum_comparison.png", dpi=150)
        plt.close()
        print("  Saved: svd_spectrum_comparison.png")


# ==========================================
# 5. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise SVD Analyzer for SFT vs RL Comparison"
    )
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # MODE 1: Compute
    p_compute = subparsers.add_parser('compute', help="Compute layer-wise SVD metrics")
    p_compute.add_argument('--base', required=True, help="Base model path")
    p_compute.add_argument('--sft', required=True, help="SFT model path")
    p_compute.add_argument('--rl', required=True, help="RL model path")
    p_compute.add_argument('--output', default='layerwise_svd.json', help="Output JSON path")
    p_compute.add_argument('--llm', action='store_true', help="Use AutoModelForCausalLM (LLM mode)")
    p_compute.add_argument('--topk', type=int, default=50, help="Top-k singular values to store")
    p_compute.add_argument('--weight-filter', type=str, default=None,
                          help="Filter weights by name (e.g., 'down_proj'). None = all weights.")

    # MODE 2: Plot
    p_plot = subparsers.add_parser('plot', help="Generate plots from computed data")
    p_plot.add_argument('--input', required=True, help="Input JSON from compute mode")
    p_plot.add_argument('--output-dir', default='svd_plots', help="Output directory for plots")

    args = parser.parse_args()

    if args.mode == 'compute':
        analyzer = LayerwiseSVDAnalyzer(
            base_path=args.base,
            sft_path=args.sft,
            rl_path=args.rl,
            is_llm=args.llm,
            topk=args.topk,
            weight_filter=args.weight_filter,
        )
        results = analyzer.run()

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(results, out_path)
        print(f"\nSaved results to: {out_path}")

    elif args.mode == 'plot':
        data = load_json(args.input)
        viz = LayerwiseSVDVisualizer(args.output_dir)
        viz.plot_all(data)


if __name__ == "__main__":
    main()
