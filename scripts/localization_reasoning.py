"""
MATH-500 / GSM8K Zero Ablation Evaluation Script
=================================================
Layer-wise zero ablation: set attention + MLP output to zero, keep skip connection.
For each layer i: h_{i+1} = h_i (instead of h_i + Attn(h_i) + MLP(h_i))

Supports both MATH-500 and GSM8K datasets.
"""

import torch
import numpy as np
from pathlib import Path
import json
import re
import os
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple, Callable
from datasets import load_dataset
import argparse
from dataclasses import dataclass
import time
import gc

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not installed. Using fallback answer checking.")

from transformers import AutoProcessor


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ZeroAblationResult:
    """Result from a single zero ablation evaluation."""
    id: str
    question: str
    gold_answer: str
    predicted: str
    correct: bool
    full_response: str
    model_type: str
    model: str
    ablation: str
    ablated_layer: int


# ============================================================================
# Constants
# ============================================================================

REASONING_MODEL_PATTERNS = [
    "openmmreasoner",
    "coldstart",
    "rl",
]

DEFAULT_MODEL_PATTERNS = [
    "qwen2.5-vl-7b-instruct",
    "qwen/qwen2.5-vl",
]


def detect_model_type(model_name: str) -> str:
    """Detect whether a model should use 'reasoner' or 'base' prompting."""
    name_lower = model_name.lower()

    for pattern in REASONING_MODEL_PATTERNS:
        if pattern in name_lower:
            return "reasoner"

    for pattern in DEFAULT_MODEL_PATTERNS:
        if pattern in name_lower:
            return "base"

    return "reasoner"


# ============================================================================
# Helper Functions
# ============================================================================

def sanitize_model_name(name: str) -> str:
    """Sanitize model name for use as directory/file name."""
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)


def extract_boxed_answer(predict_str: str) -> Optional[str]:
    """Extract answer from \\boxed{} format, handling nested braces."""
    if not predict_str:
        return None

    boxed_pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(boxed_pattern, predict_str))

    if not matches:
        return None

    last_match = matches[-1]
    start_idx = last_match.end()

    brace_count = 1
    idx = start_idx
    while idx < len(predict_str) and brace_count > 0:
        if predict_str[idx] == '{':
            brace_count += 1
        elif predict_str[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        return predict_str[start_idx:idx-1].strip()

    return None


def extract_answer_tag(predict_str: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags or \\boxed{}."""
    if not predict_str:
        return None

    answer_match = re.search(r'<answer>(.*?)</answer>', predict_str, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    boxed = extract_boxed_answer(predict_str)
    if boxed:
        return boxed

    return None


# ============================================================================
# Zero Ablation
# ============================================================================

def zero_ablate_layer(layer, layer_idx: int, config) -> Callable:
    """
    Zero-ablate a transformer layer by making it return input unchanged.
    This is equivalent to setting Attn + MLP outputs to zero while preserving skip connection.

    h_{i+1} = h_i (instead of h_i + Attn(LN(h_i)) + MLP(LN(h_i')))

    Returns the original forward function for restoration.
    """
    from transformers import DynamicCache

    orig_forward = layer.forward

    # Get dimensions from config
    num_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    def forward_identity(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs
    ):
        # Return hidden states unchanged, preserving HuggingFace return structure
        outputs = (hidden_states,)

        if output_attentions:
            # Return None for attention weights
            outputs += (None,)

        if use_cache:
            # Create proper KV cache entry for this layer
            batch_size, seq_len, _ = hidden_states.shape

            if past_key_value is not None:
                # Update cache with zeros for new tokens
                if cache_position is not None:
                    new_seq_len = cache_position.shape[0]
                else:
                    new_seq_len = seq_len

                # Create zero key/value for new positions
                # Shape must be (batch_size, num_heads, seq_len, head_dim) to match HF cache format
                zero_k = torch.zeros(
                    batch_size, num_heads, new_seq_len, head_dim,
                    dtype=hidden_states.dtype, device=hidden_states.device
                )
                zero_v = torch.zeros(
                    batch_size, num_heads, new_seq_len, head_dim,
                    dtype=hidden_states.dtype, device=hidden_states.device
                )

                # Update the cache
                past_key_value.update(zero_k, zero_v, layer_idx)
                outputs += (past_key_value,)
            else:
                outputs += (None,)

        return outputs

    layer.forward = forward_identity
    return orig_forward


def restore_layer(layer, orig_forward: Callable):
    """Restore a layer's original forward function."""
    layer.forward = orig_forward


# ============================================================================
# Dataset
# ============================================================================

class MATH500Dataset:
    """Load MATH-500 dataset (text-only)."""

    def __init__(self, split: str = "test"):
        print(f"Loading HuggingFaceH4/MATH-500 ({split})...")
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
        print(f"  Loaded {len(self.dataset)} samples")

    def get_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        return {
            "id": str(item.get("unique_id", idx)),
            "idx": idx,
            "question": item.get("problem", ""),
            "answer": str(item.get("answer", "")),
        }

    def __len__(self):
        return len(self.dataset)


class GSM8KDataset:
    """Load GSM8K dataset (text-only)."""

    def __init__(self, split: str = "test"):
        print(f"Loading openai/gsm8k ({split})...")
        self.dataset = load_dataset("openai/gsm8k", "main", split=split)
        print(f"  Loaded {len(self.dataset)} samples")

    def get_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        # GSM8K answer format: "#### 42" (extract the number after ####)
        answer_str = str(item.get("answer", ""))
        # Extract the final answer after ####
        answer_match = re.search(r'####\s*(.+)', answer_str)
        if answer_match:
            answer = answer_match.group(1).strip().replace(",", "")
        else:
            answer = answer_str.strip()

        return {
            "id": str(idx),
            "idx": idx,
            "question": item.get("question", ""),
            "answer": answer,
        }

    def __len__(self):
        return len(self.dataset)


def subsample_indices(n: int, k: int, seed: int) -> List[int]:
    """Random subsample k indices from n with fixed seed."""
    rng = np.random.RandomState(seed)
    if k >= n:
        return list(range(n))
    return sorted(rng.choice(np.arange(n), size=k, replace=False).tolist())


# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPT_REASONER = """You are a helpful assistant. When the user asks a question, your response must include two parts: first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."""

SYSTEM_PROMPT_DEFAULT = None

QUERY_PROMPT_PREFIX = 'Please solve the problem step by step and put your answer in one "\\boxed{}".\n\n'


# ============================================================================
# Evaluator
# ============================================================================

class HFEvaluator:
    """HuggingFace-based evaluation for MATH-500."""

    def __init__(
        self,
        model,
        processor,
        model_type: str = "reasoner",
        max_new_tokens: int = 8096,
        temperature: float = 0.0,
    ):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _get_system_prompt(self) -> Optional[str]:
        if self.model_type == "reasoner":
            return SYSTEM_PROMPT_REASONER
        return SYSTEM_PROMPT_DEFAULT

    def evaluate_single(self, item: Dict) -> Optional[Dict]:
        """Evaluate a single MATH-500 item."""
        system_prompt = self._get_system_prompt()
        question_text = QUERY_PROMPT_PREFIX + item["question"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question_text})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor.tokenizer([text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response = self.processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        predicted = self._extract_answer(response)
        correct = self._check_answer(predicted, item["answer"])

        result = {
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "predicted": predicted,
            "correct": correct,
            "full_response": response,
            "model_type": self.model_type,
        }

        del inputs, outputs
        torch.cuda.empty_cache()

        return result

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        if not response:
            return ""

        # Try <answer> tags first
        extracted = extract_answer_tag(response)
        if extracted:
            return extracted.strip()

        # Try \boxed{}
        boxed = extract_boxed_answer(response)
        if boxed:
            return boxed

        # Fallback patterns
        patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*[:\s]*(.+?)(?:\.|$)',
            r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(.+?)(?:\.|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                if result and len(result) < 100:
                    return result

        # Last number
        numbers = re.findall(r'(-?\d+\.?\d*)', response)
        if numbers:
            return numbers[-1]

        return response.strip()[-50:] if response else ""

    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold."""
        if not predicted or not gold:
            return False

        pred = predicted.strip()
        gold_clean = gold.strip()
        pred_norm = pred.lower().replace(" ", "")
        gold_norm = gold_clean.lower().replace(" ", "")

        # Exact match
        if pred_norm == gold_norm:
            return True

        # Math verify
        if MATH_VERIFY_AVAILABLE:
            try:
                if verify(parse(pred), parse(gold_clean)):
                    return True
            except:
                pass

        # Numeric comparison
        try:
            pred_num = float(pred_norm.replace(",", "").replace("%", ""))
            gold_num = float(gold_norm.replace(",", "").replace("%", ""))
            if abs(pred_num - gold_num) < 1e-6:
                return True
            if gold_num != 0 and abs(pred_num - gold_num) / abs(gold_num) < 0.01:
                return True
        except:
            pass

        return False


# ============================================================================
# Model Manager with Zero Ablation Support
# ============================================================================

class ModelManager:
    """Manage model loading with zero ablation support."""

    def __init__(self):
        self.processor = None
        self.current_model = None
        self.current_model_path = None

    def _get_layers(self, model):
        """Get transformer layers from model."""
        # Qwen2.5-VL (newer transformers): model.model.language_model.layers
        if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            return model.model.language_model.layers
        # Qwen2.5-VL alternative: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        # Some nested architectures: model.model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
            return model.model.model.layers
        # Qwen2.5 text-only (AutoModelForCausalLM): model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Cannot find layers")

    def get_num_layers(self, model) -> int:
        """Get the number of transformer layers."""
        return len(self._get_layers(model))

    def load_model(self, model_path: str):
        """Load a model (reuse if already loaded)."""
        if self.current_model is not None and self.current_model_path == model_path:
            return self.current_model

        # Cleanup previous model
        if self.current_model is not None:
            del self.current_model
            torch.cuda.empty_cache()
            gc.collect()

        print(f"  Loading model: {model_path}")
        from transformers import Qwen2_5_VLForConditionalGeneration

        # Use single GPU to avoid multi-device issues
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        ).eval()

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.current_model = model
        self.current_model_path = model_path
        return model

    def apply_zero_ablation(self, model, layer_idx: int) -> Callable:
        """Apply zero ablation to a specific layer. Returns restore function."""
        layers = self._get_layers(model)
        # Get the language model config for KV cache dimensions
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            config = model.model.language_model.config
        elif hasattr(model, "config"):
            config = model.config
        else:
            config = model.model.config
        orig_forward = zero_ablate_layer(layers[layer_idx], layer_idx, config)
        return lambda: restore_layer(layers[layer_idx], orig_forward)

    def cleanup(self):
        """Cleanup model."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_path = None
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# Zero Ablation Experiment
# ============================================================================

class ZeroAblationExperiment:
    """
    Manages zero ablation experiments for reasoning tasks (MATH-500, GSM8K).

    This class encapsulates the entire experimental workflow:
    - Loading datasets and models
    - Applying layer-wise zero ablation
    - Running evaluations
    - Collecting and saving results
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        dataset_name: str = "math500",
        run_name: Optional[str] = None,
        sample_size: int = 100,
        sample_seed: int = 123,
        max_new_tokens: int = 2048,
        layers: Optional[List[int]] = None,
    ):
        """Initialize the zero ablation experiment."""
        self.model_path = model_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name.lower()
        self.sample_size = sample_size
        self.sample_seed = sample_seed
        self.max_new_tokens = max_new_tokens
        self.layers_to_test = layers

        # Validate dataset
        if self.dataset_name not in ["math500", "gsm8k"]:
            raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'math500' or 'gsm8k'")

        # Setup run directory: output_dir/{model_name}/{dataset_name}
        model_name = sanitize_model_name(Path(model_path).name)
        self.run_name = run_name or model_name
        self.run_dir = Path(output_dir) / self.run_name / self.dataset_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)
        (self.run_dir / "splits").mkdir(exist_ok=True)

        # Initialize components
        self.model_manager = ModelManager()
        self.model = None
        self.model_type = None
        self.num_layers = None
        self.dataset = None
        self.evaluator = None
        self.all_results = {}

    def setup(self):
        """Setup model, dataset, and experiment configuration."""
        # Load model
        print(f"\n{'='*60}")
        print(f"Loading Model: {self.model_path}")
        print(f"{'='*60}")

        self.model = self.model_manager.load_model(self.model_path)
        self.model_type = detect_model_type(self.model_path)
        self.num_layers = self.model_manager.get_num_layers(self.model)

        print(f"  Model type: {self.model_type}")
        print(f"  Number of layers: {self.num_layers}")

        # Determine layers to ablate
        if self.layers_to_test is None:
            self.layers_to_ablate = list(range(self.num_layers))
        else:
            self.layers_to_ablate = [l for l in self.layers_to_test if 0 <= l < self.num_layers]

        # Load dataset
        print(f"\n{'='*60}")
        if self.dataset_name == "math500":
            print("Loading MATH-500 Dataset")
            self.dataset = MATH500Dataset(split="test")
        else:  # gsm8k
            print("Loading GSM8K Dataset")
            self.dataset = GSM8KDataset(split="test")
        print(f"{'='*60}")

        # Create evaluator
        self.evaluator = HFEvaluator(
            model=self.model,
            processor=self.model_manager.processor,
            model_type=self.model_type,
            max_new_tokens=self.max_new_tokens,
        )

        # Save configuration
        config = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "layers_to_ablate": self.layers_to_ablate,
            "sample_size": self.sample_size,
            "sample_seed": self.sample_seed,
            "max_new_tokens": self.max_new_tokens,
            "dataset": self.dataset_name,
        }
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def _get_sample_indices(self) -> List[int]:
        """Get or create sample indices for evaluation."""
        all_ids = [str(self.dataset.get_item(i)["id"]) for i in range(len(self.dataset))]
        id_map = {sid: i for i, sid in enumerate(all_ids)}

        split_path = self.run_dir / "splits" / f"{self.dataset_name}_seed{self.sample_seed}_n{self.sample_size}.jsonl"
        split_path.parent.mkdir(parents=True, exist_ok=True)

        if split_path.exists():
            chosen_ids = [json.loads(l)["id"] for l in split_path.read_text().splitlines() if l.strip()]
        else:
            idxs = subsample_indices(len(all_ids), self.sample_size, self.sample_seed)
            chosen_ids = [all_ids[i] for i in idxs]
            with open(split_path, "w") as f:
                for _id in chosen_ids:
                    f.write(json.dumps({"id": _id}) + "\n")

        indices = [id_map[sid] for sid in chosen_ids]
        print(f"Evaluating {len(indices)} samples per configuration")
        return indices

    def _evaluate_configuration(
        self,
        exp_name: str,
        layer_idx: int,
        indices: List[int]
    ) -> Dict[str, Any]:
        """Evaluate a single configuration (baseline or specific layer ablation)."""
        print(f"\n{'='*60}")
        if layer_idx == -1:
            print(f"Evaluating: Baseline (no ablation)")
        else:
            print(f"Evaluating: Zero ablation layer {layer_idx}")
        print(f"{'='*60}")

        # Apply zero ablation if not baseline
        restore_fn = None
        if layer_idx >= 0:
            restore_fn = self.model_manager.apply_zero_ablation(self.model, layer_idx)
            print(f"  Applied zero ablation to layer {layer_idx}")

        # Setup output
        exp_dir = self.run_dir / "results" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        out_path = exp_dir / f"{self.dataset_name}.jsonl"

        # Load existing results
        done = {}
        if out_path.exists():
            for ln in out_path.read_text().splitlines():
                if ln.strip():
                    try:
                        rec = json.loads(ln)
                        done[str(rec["id"])] = rec
                    except:
                        pass
            print(f"  Loaded {len(done)} existing results")

        # Evaluate samples
        correct_count = 0
        total_count = 0

        with open(out_path, "a") as fout:
            for ix in tqdm(indices, desc=f"  {exp_name}"):
                item = self.dataset.get_item(ix)
                sid = str(item["id"])

                if sid in done:
                    if done[sid].get("correct", False):
                        correct_count += 1
                    total_count += 1
                    continue

                result_dict = self.evaluator.evaluate_single(item)
                if result_dict is None:
                    continue

                # Add metadata
                result_dict["model"] = self.model_path
                result_dict["ablation"] = exp_name
                result_dict["ablated_layer"] = layer_idx

                fout.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                fout.flush()

                if result_dict["correct"]:
                    correct_count += 1
                total_count += 1

        # Calculate accuracy
        acc = correct_count / total_count if total_count > 0 else 0.0
        print(f"  Accuracy: {acc:.2%} ({correct_count}/{total_count})")

        # Save summary
        summary = {
            "experiment": exp_name,
            "ablated_layer": layer_idx,
            "accuracy": round(acc, 4),
            "correct": correct_count,
            "total": total_count,
        }
        (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # Restore layer if ablated
        if restore_fn is not None:
            restore_fn()
            print(f"  Restored layer {layer_idx}")

        return summary

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Run the complete zero ablation experiment."""
        # Get sample indices
        indices = self._get_sample_indices()

        # Create experiment configurations: baseline + each layer
        experiment_configs = [("baseline", -1)] + [
            (f"layer_{i}", i) for i in self.layers_to_ablate
        ]

        # Run each configuration
        for exp_name, layer_idx in experiment_configs:
            summary = self._evaluate_configuration(exp_name, layer_idx, indices)
            self.all_results[exp_name] = summary

        # Save overall summary
        (self.run_dir / "all_results.json").write_text(
            json.dumps(self.all_results, indent=2)
        )

        return self.all_results

    def print_summary(self):
        """Print experiment summary table."""
        print(f"\n{'='*60}")
        print("Experiment Complete")
        print(f"Results saved to: {self.run_dir}")
        print(f"{'='*60}")

        print("\nSummary:")
        print("-" * 50)
        baseline_acc = self.all_results.get("baseline", {}).get("accuracy", 0)
        print(f"  {'Baseline':<20} {baseline_acc:.2%}")
        print("-" * 50)

        for exp_name, res in self.all_results.items():
            if exp_name == "baseline":
                continue
            diff = res['accuracy'] - baseline_acc
            diff_str = f"{diff:+.2%}" if diff != 0 else "  0.00%"
            print(f"  {exp_name:<20} {res['accuracy']:.2%} ({diff_str})")
        print("-" * 50)

    def cleanup(self):
        """Cleanup resources."""
        self.model_manager.cleanup()


# ============================================================================
# Main
# ============================================================================

def run_zero_ablation_experiment(
    model_path: str,
    output_dir: str,
    run_name: str,
    sample_size: int,
    sample_seed: int,
    max_new_tokens: int,
    layers: Optional[List[int]] = None,
    dataset_name: str = "math500",
):
    """
    Run zero ablation experiment on MATH-500 or GSM8K.

    This is a convenience wrapper around the ZeroAblationExperiment class.

    For each layer:
    1. Apply zero ablation (layer returns input unchanged)
    2. Evaluate on dataset
    3. Restore layer
    4. Move to next layer

    Args:
        dataset_name: "math500" or "gsm8k"
    """
    experiment = ZeroAblationExperiment(
        model_path=model_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        run_name=run_name,
        sample_size=sample_size,
        sample_seed=sample_seed,
        max_new_tokens=max_new_tokens,
        layers=layers,
    )

    experiment.setup()
    experiment.run()
    experiment.print_summary()
    experiment.cleanup()


def main():
    """Main entry point for zero ablation experiments."""
    parser = argparse.ArgumentParser(
        description="Layer-wise Zero Ablation Evaluation for Reasoning Tasks"
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="Model path (e.g., Qwen/Qwen2.5-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--dataset", type=str, default="math500",
        choices=["math500", "gsm8k", "all"],
        help="Dataset to evaluate on (default: math500)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/localization_reasoning",
        help="Output directory for results (default: ./results/localization_reasoning)"
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Custom run name (default: auto-generated)"
    )
    parser.add_argument(
        "--sample_size", type=int, default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--sample_seed", type=int, default=123,
        help="Random seed for sampling (default: 123)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Specific layers to ablate (default: all layers)"
    )

    args = parser.parse_args()

    # Determine which datasets to run
    if args.dataset == "all":
        datasets = ["math500", "gsm8k"]
    else:
        datasets = [args.dataset]

    # Run experiment for each dataset
    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset_name.upper()}")
        print(f"{'#'*60}")

        experiment = ZeroAblationExperiment(
            model_path=args.model,
            output_dir=args.output_dir,
            dataset_name=dataset_name,
            run_name=args.run_name,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            max_new_tokens=args.max_new_tokens,
            layers=args.layers,
        )

        experiment.setup()
        experiment.run()
        experiment.print_summary()
        experiment.cleanup()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
