"""
Layer-wise Visual Token Swapping Evaluation Script

Evaluates the change rate of model outputs when swapping visual tokens
at different layers across four VFL tasks: counting, ocr, grounding, recognition.

Usage:
    python run_layer_swapping_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --num_samples 50
    python run_layer_swapping_eval.py --tasks counting ocr --layers 0 5 10 15 20 25
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

# Add parent and utils directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from vision_token_swapper import (
    VisualTokenSwapper,
    reset_swap_config,
    set_swap_config,
    SWAP_CONFIG,
)


TASK_DIRS = {
    "counting": Path(__file__).parent.parent / "dataset" / "vision_functionality_dataset" / "counting",
    "ocr": Path(__file__).parent.parent / "dataset" / "vision_functionality_dataset" / "ocr",
    "grounding": Path(__file__).parent.parent / "dataset" / "vision_functionality_dataset" / "grounding",
    "recognition": Path(__file__).parent.parent / "dataset" / "vision_functionality_dataset" / "recognition",
}


@dataclass
class LayerSwapResult:
    """Result from a single layer swap experiment."""
    task: str
    sample_id: str
    layer: int
    original_answer: str
    swapped_answer: str
    answer_changed: bool
    target_answer: str
    source_answer: str


def load_task_data(task: str, num_samples: Optional[int] = None) -> List[dict]:
    """Load data for a specific task."""
    task_dir = TASK_DIRS[task]
    data_file = task_dir / "data.jsonl"

    samples = []
    with open(data_file, "r") as f:
        for line in f:
            sample = json.loads(line.strip())
            # Convert relative paths to absolute
            sample["target_image"] = str(task_dir / sample["target_image"])
            sample["source_image"] = str(task_dir / sample["source_image"])
            samples.append(sample)

            if num_samples and len(samples) >= num_samples:
                break

    return samples


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Basic normalization: lowercase, strip whitespace
    answer = answer.lower().strip()
    # Remove punctuation at the end
    answer = answer.rstrip(".,!?;:")
    return answer


def answers_are_different(ans1: str, ans2: str) -> bool:
    """Check if two answers are meaningfully different."""
    norm1 = normalize_answer(ans1)
    norm2 = normalize_answer(ans2)

    # For short answers, require exact match to be "same"
    if len(norm1) < 20 and len(norm2) < 20:
        return norm1 != norm2

    # For longer answers, check if they share significant content
    # (simple heuristic: if one is substring of other, consider same)
    if norm1 in norm2 or norm2 in norm1:
        return False

    return norm1 != norm2


def run_layer_swapping_evaluation(
    swapper: VisualTokenSwapper,
    tasks: List[str],
    layers: List[int],
    num_samples: int,
    output_dir: Path,
) -> Dict[str, Dict[int, float]]:
    """
    Run layer-wise swapping evaluation across tasks.

    Returns:
        Dict mapping task -> layer -> change_rate
    """
    all_results = []
    change_rates = defaultdict(dict)  # task -> layer -> rate

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating task: {task.upper()}")
        print(f"{'='*60}")

        samples = load_task_data(task, num_samples)
        print(f"Loaded {len(samples)} samples")

        # Track changes per layer
        layer_changes = defaultdict(list)  # layer -> list of bool

        for sample in tqdm(samples, desc=f"Processing {task}"):
            sample_id = sample["id"]
            target_image = sample["target_image"]
            source_image = sample["source_image"]
            prompt = sample["prompt"]

            # Get target/source answers from metadata
            meta = sample.get("meta", {})
            if task == "counting":
                target_answer = meta.get("target_answer", "")
                source_answer = meta.get("source_answer", "")
            elif task == "ocr":
                target_answer = meta.get("target_text", "")
                source_answer = meta.get("source_text", "")
            elif task == "grounding":
                target_answer = str(meta.get("target_bbox_xyxy", ""))
                source_answer = str(meta.get("source_bbox_xyxy", ""))
            elif task == "recognition":
                target_answer = meta.get("target_answer", "")
                source_answer = meta.get("source_answer", "")
            else:
                target_answer = ""
                source_answer = ""

            # Get original answer (no swapping)
            try:
                original_answer = swapper._generate_original(
                    target_image, prompt, max_new_tokens=64
                )
            except Exception as e:
                print(f"  Error on {sample_id} (original): {e}")
                continue

            # Collect source KV cache once per sample
            try:
                source_kv_cache, src_vision_start, src_vision_length = \
                    swapper._collect_source_kv(source_image, prompt)
            except Exception as e:
                print(f"  Error on {sample_id} (collect KV): {e}")
                continue

            # Test each layer
            for layer in layers:
                try:
                    swapped_answer = swapper._generate_with_swap(
                        target_image,
                        prompt,
                        swap_layer=layer,
                        source_kv_cache=source_kv_cache,
                        source_vision_start=src_vision_start,
                        source_vision_length=src_vision_length,
                        max_new_tokens=64,
                    )

                    changed = answers_are_different(original_answer, swapped_answer)
                    layer_changes[layer].append(changed)

                    result = LayerSwapResult(
                        task=task,
                        sample_id=sample_id,
                        layer=layer,
                        original_answer=original_answer,
                        swapped_answer=swapped_answer,
                        answer_changed=changed,
                        target_answer=target_answer,
                        source_answer=source_answer,
                    )
                    all_results.append(result)

                except Exception as e:
                    import traceback
                    print(f"\n  Error on {sample_id} layer {layer}: {e}")
                    traceback.print_exc()
                    continue

            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate change rates for this task
        print(f"\n{task.upper()} - Layer-wise Change Rates:")
        print("-" * 40)
        for layer in sorted(layers):
            if layer in layer_changes and layer_changes[layer]:
                rate = sum(layer_changes[layer]) / len(layer_changes[layer])
                change_rates[task][layer] = rate
                bar = "█" * int(rate * 30)
                print(f"  Layer {layer:2d}: {rate:6.2%} |{bar}")
            else:
                change_rates[task][layer] = 0.0
                print(f"  Layer {layer:2d}: N/A")

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "layer_swap_results.jsonl"
    with open(results_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"\nDetailed results saved to: {results_file}")

    # Save summary
    summary_file = output_dir / "layer_swap_summary.json"
    summary = {
        "tasks": tasks,
        "layers": layers,
        "num_samples": num_samples,
        "change_rates": {task: dict(rates) for task, rates in change_rates.items()},
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    return change_rates


def print_summary_table(change_rates: Dict[str, Dict[int, float]], layers: List[int]):
    """Print a summary table of change rates across tasks and layers."""
    print("\n" + "=" * 80)
    print("SUMMARY: Layer-wise Change Rates Across Tasks")
    print("=" * 80)

    # Header
    header = f"{'Layer':>6} |"
    for task in change_rates.keys():
        header += f" {task:>12} |"
    print(header)
    print("-" * len(header))

    # Rows
    for layer in sorted(layers):
        row = f"{layer:>6} |"
        for task in change_rates.keys():
            rate = change_rates[task].get(layer, 0.0)
            row += f" {rate:>11.2%} |"
        print(row)

    print("-" * len(header))

    # Find peak layers for each task
    print("\nPeak VFL Layers (highest change rate):")
    for task, rates in change_rates.items():
        if rates:
            peak_layer = max(rates.keys(), key=lambda l: rates[l])
            peak_rate = rates[peak_layer]
            print(f"  {task}: Layer {peak_layer} ({peak_rate:.2%})")


def plot_change_rates(
    change_rates: Dict[str, Dict[int, float]],
    layers: List[int],
    output_path: Path
):
    """Plot change rates for visualization."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']

        for idx, (task, rates) in enumerate(change_rates.items()):
            sorted_layers = sorted(rates.keys())
            values = [rates[l] for l in sorted_layers]
            ax.plot(sorted_layers, values,
                   label=task,
                   color=colors[idx % len(colors)],
                   marker=markers[idx % len(markers)],
                   linewidth=2,
                   markersize=6)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Change Rate", fontsize=12)
        ax.set_title("Visual Token Swapping: Layer-wise Change Rates", fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        plt.close()

    except ImportError:
        print("\nNote: matplotlib not available, skipping plot generation")


def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise Visual Token Swapping Evaluation"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        default=["counting", "ocr", "grounding", "recognition"],
        choices=["counting", "ocr", "grounding", "recognition"],
        help="Tasks to evaluate"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Specific layers to test (default: all layers)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50,
        help="Number of samples per task (default: 50)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype"
    )

    args = parser.parse_args()

    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Initialize swapper
    print(f"Loading model: {args.model}")
    swapper = VisualTokenSwapper(
        model_name=args.model,
        device="cuda",
        dtype=dtype,
    )

    # Determine layers to test
    if args.layers is not None:
        layers = args.layers
    else:
        layers = list(range(swapper.num_layers))

    print(f"Testing {len(layers)} layers: {layers}")
    print(f"Tasks: {args.tasks}")
    print(f"Samples per task: {args.num_samples}")

    # Set output directory: results/localization_vision/{model_name}
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        import re
        model_name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', Path(args.model).name)
        output_dir = Path(__file__).parent.parent / "results" / "localization_vision" / model_name

    # Run evaluation
    change_rates = run_layer_swapping_evaluation(
        swapper=swapper,
        tasks=args.tasks,
        layers=layers,
        num_samples=args.num_samples,
        output_dir=output_dir,
    )

    # Print summary
    print_summary_table(change_rates, layers)

    # Generate plot
    plot_path = output_dir / "change_rates_plot.png"
    plot_change_rates(change_rates, layers, plot_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
