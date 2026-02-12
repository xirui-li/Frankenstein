"""
Visual Token Swapping for Vision Function Layer Analysis
Based on: "Vision Function Layer in Multimodal LLMs" (NeurIPS 2025)
https://github.com/ChengShiest/Vision-Function-Layer

This script implements the visual token swapping mechanism to analyze
layer-specific visual functions in MLLMs.

Key idea: Swap vision tokens in the KV cache at specific layers between
a source image and a target image to see which layer affects which function
(e.g., OCR, counting, grounding, recognition).

Requirements:
    pip install transformers==4.50.0 torch pillow accelerate flash-attn
"""

import torch
import warnings
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from PIL import Image
import json
from pathlib import Path

import transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLFlashAttention2,
    Qwen2_5_VLModel,
    Qwen2_5_VLDecoderLayer,
    BaseModelOutputWithPast,
    _flash_attention_forward,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache


# Global variables to control swapping behavior
SWAP_CONFIG = {
    "enabled": False,
    "swap_layer": -1,  # Layer at which to swap vision tokens (-1 = no swap)
    "source_kv_cache": None,  # Cached KV states from source image
    "source_vision_start": 0,
    "source_vision_length": 0,
    "target_vision_start": 0,
    "target_vision_length": 0,
}


def reset_swap_config():
    """Reset swap configuration to default state."""
    SWAP_CONFIG["enabled"] = False
    SWAP_CONFIG["swap_layer"] = -1
    SWAP_CONFIG["source_kv_cache"] = None
    SWAP_CONFIG["source_vision_start"] = 0
    SWAP_CONFIG["source_vision_length"] = 0
    SWAP_CONFIG["target_vision_start"] = 0
    SWAP_CONFIG["target_vision_length"] = 0


def set_swap_config(
    swap_layer: int,
    source_kv_cache: dict,
    source_vision_start: int,
    source_vision_length: int,
    target_vision_start: int,
    target_vision_length: int,
):
    """Configure the swap parameters."""
    SWAP_CONFIG["enabled"] = True
    SWAP_CONFIG["swap_layer"] = swap_layer
    SWAP_CONFIG["source_kv_cache"] = source_kv_cache
    SWAP_CONFIG["source_vision_start"] = source_vision_start
    SWAP_CONFIG["source_vision_length"] = source_vision_length
    SWAP_CONFIG["target_vision_start"] = target_vision_start
    SWAP_CONFIG["target_vision_length"] = target_vision_length


# ============================================================================
# Modified Forward Functions for Visual Token Swapping
# ============================================================================

def forward_attention_with_swap(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """
    Modified attention forward that supports visual token swapping.
    
    At the specified swap_layer, replaces the vision token KV states from
    target image with those from the source image.
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # ========== VISUAL TOKEN SWAPPING ==========
    if (
        SWAP_CONFIG["enabled"]
        and self.layer_idx >= SWAP_CONFIG["swap_layer"]
        and SWAP_CONFIG["source_kv_cache"] is not None
        and self.layer_idx in SWAP_CONFIG["source_kv_cache"]
    ):
        src_kv = SWAP_CONFIG["source_kv_cache"][self.layer_idx]
        src_k, src_v = src_kv["key"], src_kv["value"]
        
        src_start = SWAP_CONFIG["source_vision_start"]
        src_len = SWAP_CONFIG["source_vision_length"]
        tgt_start = SWAP_CONFIG["target_vision_start"]
        tgt_len = SWAP_CONFIG["target_vision_length"]
        
        # Extract source vision tokens
        src_vision_k = src_k[:, :, src_start:src_start + src_len, :]
        src_vision_v = src_v[:, :, src_start:src_start + src_len, :]
        
        # Handle dimension mismatch (use interpolation or truncation)
        if src_len != tgt_len:
            # Simple approach: interpolate to match target length
            src_vision_k = torch.nn.functional.interpolate(
                src_vision_k.permute(0, 1, 3, 2),  # [B, H, D, src_len]
                size=tgt_len,
                mode='linear',
                align_corners=False
            ).permute(0, 1, 3, 2)  # [B, H, tgt_len, D]
            src_vision_v = torch.nn.functional.interpolate(
                src_vision_v.permute(0, 1, 3, 2),
                size=tgt_len,
                mode='linear',
                align_corners=False
            ).permute(0, 1, 3, 2)
        
        # Swap: replace target vision tokens with source vision tokens
        key_states = torch.cat([
            key_states[:, :, :tgt_start, :],
            src_vision_k.to(key_states.device, key_states.dtype),
            key_states[:, :, tgt_start + tgt_len:, :]
        ], dim=2)
        
        value_states = torch.cat([
            value_states[:, :, :tgt_start, :],
            src_vision_v.to(value_states.device, value_states.dtype),
            value_states[:, :, tgt_start + tgt_len:, :]
        ], dim=2)
    # ========== END VISUAL TOKEN SWAPPING ==========

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def forward_attention_collect_kv(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """
    Attention forward that collects KV states for later swapping.
    This is used during the first pass on source image.
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # Store KV states for this layer (before cache update)
    if not hasattr(self, "_collected_kv"):
        self._collected_kv = {}
    self._collected_kv[self.layer_idx] = {
        "key": key_states.detach().clone(),
        "value": value_states.detach().clone(),
    }

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# ============================================================================
# Visual Token Swapping Evaluator
# ============================================================================

@dataclass
class SwapResult:
    """Result from a single swap experiment."""
    swap_layer: int
    source_image: str
    target_image: str
    question: str
    original_answer: str
    swapped_answer: str
    ground_truth: str
    is_correct_original: bool
    is_correct_swapped: bool


class VisualTokenSwapper:
    """
    Evaluator for visual token swapping experiments.
    
    Usage:
        swapper = VisualTokenSwapper(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Run single swap experiment
        result = swapper.swap_and_evaluate(
            source_image="source.jpg",
            target_image="target.jpg",
            question="How many objects are in the image?",
            swap_layer=18,
            ground_truth="5"
        )
        
        # Run pairwise evaluation across all layers
        results = swapper.evaluate_pairwise_dataset(dataset, layers=range(28))
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded with {self.num_layers} layers")
        
        # Store original forward function
        self._original_attn_forward = Qwen2_5_VLFlashAttention2.forward

    def _get_model_layers(self):
        """Get the transformer layers from the model (handles different architectures)."""
        # Qwen2.5-VL specific: model.model.language_model.layers
        if hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'layers'):
            return self.model.model.language_model.layers
        # Standard: model.model.layers
        if hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        # Nested: model.model.model.layers
        if hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
            return self.model.model.model.layers
        # Fallback: model.layers
        if hasattr(self.model, 'layers'):
            return self.model.layers
        raise AttributeError(f"Cannot find layers in model. Checked: model.model.language_model.layers, model.model.layers, model.model.model.layers, model.layers")

    def _find_vision_token_positions(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Find the start position and length of vision tokens in input_ids."""
        image_token_id = self.model.config.image_token_id
        
        # Find positions where image tokens appear
        mask = input_ids[0] == image_token_id
        positions = torch.where(mask)[0]
        
        if len(positions) == 0:
            return 0, 0
        
        start = positions[0].item()
        length = len(positions)
        
        return start, length
    
    def _collect_source_kv(
        self,
        image: Union[str, Image.Image],
        question: str,
    ) -> Tuple[dict, int, int]:
        """
        Run forward pass on source image and collect KV states.
        
        Returns:
            kv_cache: Dict mapping layer_idx to {"key": tensor, "value": tensor}
            vision_start: Start position of vision tokens
            vision_length: Number of vision tokens
        """
        # Prepare input
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        vision_start, vision_length = self._find_vision_token_positions(inputs["input_ids"])
        
        # Temporarily replace forward to collect KV
        Qwen2_5_VLFlashAttention2.forward = forward_attention_collect_kv
        
        # Run forward pass
        with torch.no_grad():
            _ = self.model(**inputs, use_cache=False)
        
        # Collect KV states from all attention layers
        kv_cache = {}
        model_layers = self._get_model_layers()

        for layer in model_layers:
            if hasattr(layer.self_attn, "_collected_kv"):
                kv_cache.update(layer.self_attn._collected_kv)
                layer.self_attn._collected_kv = {}
        
        # Restore original forward
        Qwen2_5_VLFlashAttention2.forward = self._original_attn_forward
        
        return kv_cache, vision_start, vision_length
    
    def _generate_with_swap(
        self,
        image: Union[str, Image.Image],
        question: str,
        swap_layer: int,
        source_kv_cache: dict,
        source_vision_start: int,
        source_vision_length: int,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate answer with vision token swapping at specified layer."""
        # Prepare input
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        target_vision_start, target_vision_length = self._find_vision_token_positions(
            inputs["input_ids"]
        )
        
        # Configure swapping
        set_swap_config(
            swap_layer=swap_layer,
            source_kv_cache=source_kv_cache,
            source_vision_start=source_vision_start,
            source_vision_length=source_vision_length,
            target_vision_start=target_vision_start,
            target_vision_length=target_vision_length,
        )
        
        # Replace forward with swapping version
        Qwen2_5_VLFlashAttention2.forward = forward_attention_with_swap
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        # Restore original forward and reset config
        Qwen2_5_VLFlashAttention2.forward = self._original_attn_forward
        reset_swap_config()
        
        # Decode output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return answer
    
    def _generate_original(
        self,
        image: Union[str, Image.Image],
        question: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate answer without any swapping."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return answer
    
    def swap_and_evaluate(
        self,
        source_image: Union[str, Image.Image],
        target_image: Union[str, Image.Image],
        question: str,
        swap_layer: int,
        ground_truth: str = None,
        max_new_tokens: int = 128,
    ) -> SwapResult:
        """
        Perform visual token swapping experiment.
        
        Args:
            source_image: Image whose vision tokens will be swapped INTO target
            target_image: Image that will receive vision tokens from source
            question: Question to ask about the image
            swap_layer: Layer at which to perform the swap (0-indexed)
            ground_truth: Expected answer for accuracy checking
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            SwapResult with original and swapped answers
        """
        # Step 1: Get original answer (no swapping)
        original_answer = self._generate_original(
            target_image, question, max_new_tokens
        )
        
        # Step 2: Collect KV cache from source image
        source_kv_cache, src_vision_start, src_vision_length = self._collect_source_kv(
            source_image, question
        )
        
        # Step 3: Generate with swapping
        swapped_answer = self._generate_with_swap(
            target_image,
            question,
            swap_layer,
            source_kv_cache,
            src_vision_start,
            src_vision_length,
            max_new_tokens,
        )
        
        # Check correctness if ground truth provided
        is_correct_original = False
        is_correct_swapped = False
        if ground_truth is not None:
            gt_lower = ground_truth.lower().strip()
            is_correct_original = gt_lower in original_answer.lower()
            is_correct_swapped = gt_lower in swapped_answer.lower()
        
        return SwapResult(
            swap_layer=swap_layer,
            source_image=str(source_image),
            target_image=str(target_image),
            question=question,
            original_answer=original_answer,
            swapped_answer=swapped_answer,
            ground_truth=ground_truth or "",
            is_correct_original=is_correct_original,
            is_correct_swapped=is_correct_swapped,
        )
    
    def evaluate_pairwise_dataset(
        self,
        dataset: List[dict],
        layers: List[int] = None,
        output_path: str = None,
    ) -> List[SwapResult]:
        """
        Evaluate visual token swapping across multiple layers on a pairwise dataset.
        
        Args:
            dataset: List of dicts with keys:
                - source_image: path to source image
                - target_image: path to target image  
                - question: question to ask
                - ground_truth: expected answer
                - function_type: (optional) e.g., "counting", "ocr", "recognition"
            layers: List of layer indices to test (default: all layers)
            output_path: Optional path to save results as JSON
            
        Returns:
            List of SwapResult objects
        """
        if layers is None:
            layers = list(range(self.num_layers))
        
        all_results = []
        
        for idx, sample in enumerate(dataset):
            print(f"\nProcessing sample {idx + 1}/{len(dataset)}")
            
            source_image = sample["source_image"]
            target_image = sample["target_image"]
            question = sample["question"]
            ground_truth = sample.get("ground_truth")
            
            for layer in layers:
                print(f"  Testing layer {layer}/{self.num_layers - 1}...", end=" ")
                
                result = self.swap_and_evaluate(
                    source_image=source_image,
                    target_image=target_image,
                    question=question,
                    swap_layer=layer,
                    ground_truth=ground_truth,
                )
                
                all_results.append(result)
                
                # Quick summary
                status = "✓" if result.is_correct_swapped else "✗"
                print(f"{status} (orig: {result.original_answer[:30]}... -> swap: {result.swapped_answer[:30]}...)")
        
        # Save results
        if output_path:
            results_dict = [
                {
                    "swap_layer": r.swap_layer,
                    "source_image": r.source_image,
                    "target_image": r.target_image,
                    "question": r.question,
                    "original_answer": r.original_answer,
                    "swapped_answer": r.swapped_answer,
                    "ground_truth": r.ground_truth,
                    "is_correct_original": r.is_correct_original,
                    "is_correct_swapped": r.is_correct_swapped,
                }
                for r in all_results
            ]
            with open(output_path, "w") as f:
                json.dump(results_dict, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        return all_results
    
    def analyze_vfl_layers(
        self,
        results: List[SwapResult],
        function_type: str = None,
    ) -> dict:
        """
        Analyze which layers are Vision Function Layers for a given function.
        
        A layer is considered a VFL if swapping at that layer changes the answer
        to match the source image's characteristics.
        """
        layer_stats = {}
        
        for layer in range(self.num_layers):
            layer_results = [r for r in results if r.swap_layer == layer]
            
            if not layer_results:
                continue
            
            # Count how often swapping changed the answer
            changes = sum(
                1 for r in layer_results
                if r.original_answer.lower().strip() != r.swapped_answer.lower().strip()
            )
            
            layer_stats[layer] = {
                "total_samples": len(layer_results),
                "answer_changes": changes,
                "change_rate": changes / len(layer_results) if layer_results else 0,
            }
        
        # Find peak VFL layers (highest change rate)
        if layer_stats:
            sorted_layers = sorted(
                layer_stats.items(),
                key=lambda x: x[1]["change_rate"],
                reverse=True
            )
            
            return {
                "layer_stats": layer_stats,
                "top_vfl_layers": [l[0] for l in sorted_layers[:3]],
                "function_type": function_type,
            }
        
        return {"layer_stats": {}, "top_vfl_layers": [], "function_type": function_type}


# ============================================================================
# Example Usage and Demo
# ============================================================================

def create_demo_dataset():
    """Create a small demo dataset for testing."""
    return [
        {
            "source_image": "source1.jpg",  # Image with 3 objects
            "target_image": "target1.jpg",  # Image with 5 objects
            "question": "How many objects are in this image?",
            "ground_truth": "5",
            "function_type": "counting",
        },
        {
            "source_image": "source2.jpg",  # Image with text "Hello"
            "target_image": "target2.jpg",  # Image with text "World"
            "question": "What text is shown in this image?",
            "ground_truth": "World",
            "function_type": "ocr",
        },
    ]


def main():
    """Demo usage of the visual token swapper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual Token Swapping Evaluation")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--source-image", type=str, required=True,
        help="Path to source image"
    )
    parser.add_argument(
        "--target-image", type=str, required=True,
        help="Path to target image"
    )
    parser.add_argument(
        "--question", type=str, required=True,
        help="Question to ask"
    )
    parser.add_argument(
        "--swap-layer", type=int, default=-1,
        help="Layer to swap at (-1 to test all layers)"
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Ground truth answer for accuracy"
    )
    parser.add_argument(
        "--output", type=str, default="swap_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Initialize swapper
    swapper = VisualTokenSwapper(model_name=args.model)
    
    if args.swap_layer >= 0:
        # Single layer evaluation
        result = swapper.swap_and_evaluate(
            source_image=args.source_image,
            target_image=args.target_image,
            question=args.question,
            swap_layer=args.swap_layer,
            ground_truth=args.ground_truth,
        )
        
        print("\n" + "=" * 60)
        print("VISUAL TOKEN SWAPPING RESULT")
        print("=" * 60)
        print(f"Swap Layer: {result.swap_layer}")
        print(f"Question: {result.question}")
        print(f"Original Answer: {result.original_answer}")
        print(f"Swapped Answer: {result.swapped_answer}")
        if result.ground_truth:
            print(f"Ground Truth: {result.ground_truth}")
            print(f"Correct (original): {result.is_correct_original}")
            print(f"Correct (swapped): {result.is_correct_swapped}")
    else:
        # All layers evaluation
        dataset = [{
            "source_image": args.source_image,
            "target_image": args.target_image,
            "question": args.question,
            "ground_truth": args.ground_truth,
        }]
        
        results = swapper.evaluate_pairwise_dataset(
            dataset,
            layers=list(range(swapper.num_layers)),
            output_path=args.output,
        )
        
        # Analyze VFL layers
        analysis = swapper.analyze_vfl_layers(results)
        
        print("\n" + "=" * 60)
        print("VFL LAYER ANALYSIS")
        print("=" * 60)
        print(f"Top VFL Layers: {analysis['top_vfl_layers']}")
        print("\nLayer-wise change rates:")
        for layer, stats in sorted(analysis["layer_stats"].items()):
            print(f"  Layer {layer:2d}: {stats['change_rate']:.2%} ({stats['answer_changes']}/{stats['total_samples']})")


if __name__ == "__main__":
    main()