"""
VLM Evaluation: Merged Script
==============================

This merged script combines functionality from:
1. evaluate_new.py - Supports GPT, Claude, and HuggingFace models
2. evaluate_v2_revisual.py - Supports Revisual-R1 models with finer-grained splits

Supported Model Types:
- OpenAI GPT models (gpt-4o, gpt-5, o1, o3, etc.)
- Anthropic Claude models (claude-3, claude-4, etc.)
- HuggingFace models (Qwen2.5-VL, OpenMMReasoner, Revisual-R1, MMR1, etc.)

Experiment Types:
- v1 (general): General VQA + Math VQA + MATH-500

Features:
- Layer splicing, ablation, and interpolation
- GPT-based image descriptions and answer grading
- Result caching and reuse from previous runs
- Inference timeout handling
"""

import torch
import numpy as np
from pathlib import Path
import json
import re
import os
import shutil
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple
from datasets import load_dataset
from PIL import Image
import base64
import io
import argparse
from dataclasses import dataclass, field
import time
import gc
import signal
from contextlib import contextmanager

# ============================================================================
# Timeout Utilities
# ============================================================================

class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass


@contextmanager
def timeout_context(seconds: int, msg: str = "Operation timed out"):
    """Context manager that raises TimeoutException after specified seconds."""
    def timeout_handler(signum, frame):
        raise TimeoutException(msg)

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# Optional imports with availability flags
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. GPT features disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Claude features disabled.")

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not installed. Using fallback answer checking.")

from transformers import AutoProcessor


# ============================================================================
# Constants for Model Type Detection
# ============================================================================

# Models that should use REASONING prompting (with <think>/<answer> tags)
REASONING_MODEL_PATTERNS = [
    "openmmreasoner", "coldstart", "rl",
    "coldstart early", "rl early", "coldstart late", "rl late",
    "mmr1",
]

# Models that should use REVISUAL prompting (with <think> + \boxed{})
REVISUAL_MODEL_PATTERNS = [
    "revisual-r1", "revisual_r1",
]

# Models that should use DEFAULT prompting (no special system prompt)
DEFAULT_MODEL_PATTERNS = [
    "qwen2.5-vl-7b-instruct", "qwen/qwen2.5-vl", "replaced by",
]

# OpenAI GPT models (use OpenAI API instead of HuggingFace)
GPT_MODEL_PATTERNS = [
    "gpt-4", "gpt-5", "gpt-4o", "gpt-4-turbo", "gpt-4-vision", "o1", "o3", "o4",
]

# Anthropic Claude models (use Anthropic API)
CLAUDE_MODEL_PATTERNS = [
    "claude-3", "claude-4", "claude-opus", "claude-sonnet", "claude-haiku",
]


def is_gpt_model(model_name: str) -> bool:
    """Check if a model is an OpenAI GPT model."""
    name_lower = model_name.lower()
    for pattern in GPT_MODEL_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def is_claude_model(model_name: str) -> bool:
    """Check if a model is an Anthropic Claude model."""
    name_lower = model_name.lower()
    for pattern in CLAUDE_MODEL_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def detect_model_type(model_name: str) -> str:
    """
    Detect model type for appropriate prompting strategy.
    Returns: 'gpt', 'claude', 'revisual', 'reasoner', or 'base'
    """
    name_lower = model_name.lower()

    # Check for GPT models first
    if is_gpt_model(model_name):
        return "gpt"

    # Check for Claude models
    if is_claude_model(model_name):
        return "claude"

    # Check for Revisual-R1 models
    for pattern in REVISUAL_MODEL_PATTERNS:
        if pattern in name_lower:
            return "revisual"

    # Check for reasoning patterns
    for pattern in REASONING_MODEL_PATTERNS:
        if pattern in name_lower:
            if "replaced by" in name_lower:
                return "base"
            return "reasoner"

    # Check for default/base patterns
    for pattern in DEFAULT_MODEL_PATTERNS:
        if pattern in name_lower:
            return "base"

    # Default to reasoner for unknown models
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

    lines = predict_str.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            num_match = re.search(r'(-?\d+\.?\d*)\s*$', line)
            if num_match:
                return num_match.group(1)

    return None


def parse_mcq(predict_str: str) -> Optional[str]:
    """Parse multiple choice answer. Returns uppercase letter A-H or None."""
    if not predict_str:
        return None

    predict_str = predict_str.strip()

    patterns = [
        (r'<answer>\s*([A-H])\s*</answer>', re.IGNORECASE),
        (r'\\boxed\{\s*([A-H])\s*\}', re.IGNORECASE),
        (r'[Tt]he\s+(?:correct\s+)?answer\s+is\s*:?\s*\(?([A-H])\)?', 0),
        (r'[Aa]nswer\s*:\s*\(?([A-H])\)?', 0),
        (r'\(([A-H])\)\s*$', 0),
        (r'^([A-H])\.$', re.MULTILINE),
        (r'\*\*([A-H])\*\*', 0),
        (r'(?:option|choice)\s+([A-H])', re.IGNORECASE),
        (r'([A-H])\s*(?:is|would be)\s+(?:the\s+)?(?:correct|right|best)', re.IGNORECASE),
        (r'(?:select|choose)\s+([A-H])', re.IGNORECASE),
    ]

    for pattern, flags in patterns:
        match = re.search(pattern, predict_str, flags)
        if match:
            return match.group(1).upper()

    end_portion = predict_str[-100:] if len(predict_str) > 100 else predict_str
    letter_match = re.search(r'\b([A-H])\b', end_portion)
    if letter_match:
        return letter_match.group(1).upper()

    return None


def resize_image_if_needed(image: Image.Image, max_pixels: int = 1280 * 720, min_size: int = 28) -> Image.Image:
    """Resize image if it exceeds max_pixels or is smaller than min_size."""
    if image is None:
        return None

    width, height = image.size

    # First, ensure minimum dimensions (required by Qwen2.5-VL)
    if width < min_size or height < min_size:
        scale_up = max(min_size / width, min_size / height)
        width = int(width * scale_up)
        height = int(height * scale_up)
        image = image.resize((width, height), Image.LANCZOS)

    current_pixels = width * height

    if current_pixels <= max_pixels:
        return image

    scale = (max_pixels / current_pixels) ** 0.5
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.LANCZOS)


def make_black_image(img: Image.Image) -> Image.Image:
    """Return a same-size black RGB image."""
    if img is None:
        return None
    w, h = img.size
    return Image.new('RGB', (w, h), (0, 0, 0))


# ============================================================================
# GPT Helper Class
# ============================================================================

class GPTHelper:
    """OpenAI helper for image descriptions and answer grading."""

    IMAGE_DESCRIPTION_SYSTEM = """You are a precise visual analyzer for mathematical and scientific images.
Your task is to describe EXACTLY what you see - do NOT solve any problems or make inferences.

Focus on:
1. All text, numbers, labels, and symbols visible in the image
2. Geometric shapes, graphs, diagrams, or charts with their properties
3. Spatial relationships between elements
4. Colors and visual distinctions that may be meaningful
5. Any tables, axes labels, or data points

Be thorough but factual. Include ALL numbers and text you can read."""

    IMAGE_DESCRIPTION_USER = """Describe this image in detail for someone who cannot see it.
Output a structured description covering:
- Overview: What type of image is this (graph, diagram, geometry problem, etc.)
- Text Content: All readable text, numbers, labels, equations
- Visual Elements: Shapes, lines, colors, symbols
- Spatial Layout: How elements are arranged
- Key Details: Any numbers, measurements, or specific values shown

Be precise and complete. Do not solve or interpret - just describe what you see."""

    GRADING_SYSTEM = """You are a strict but fair math grader. Compare the model's answer to the ground truth.

Consider correct if:
- Numerically equivalent (e.g., 0.5 = 1/2 = 50%)
- Same meaning with different formatting (e.g., "2x+3" vs "3+2x")
- For multiple choice: letter matches or full option text matches
- Minor differences in units or notation that don't change the answer

Consider incorrect if:
- Different numerical values
- Wrong multiple choice option
- Missing or extra significant information"""

    def __init__(self, model: str = "gpt-4o-mini", cache_dir: str = "./cache"):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed")
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.desc_dir = self.cache_dir / "descriptions"
        self.grade_dir = self.cache_dir / "grading"
        self.desc_dir.mkdir(parents=True, exist_ok=True)
        self.grade_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI()

    @staticmethod
    def _image_to_data_url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _safe_key(self, key: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", key)[:200]

    def describe_image(self, cache_key: str, img: Image.Image, timeout_seconds: int = 120) -> str:
        """Generate image description using GPT."""
        p = self.desc_dir / f"{self._safe_key(cache_key)}.json"
        if p.exists():
            return json.loads(p.read_text()).get("description", "")

        data_url = self._image_to_data_url(img)

        try:
            with timeout_context(timeout_seconds, f"GPT describe_image timed out after {timeout_seconds}s"):
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.IMAGE_DESCRIPTION_SYSTEM},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": self.IMAGE_DESCRIPTION_USER},
                        ]},
                    ],
                    max_completion_tokens=2000,
                    timeout=60.0,
                )
                out = resp.choices[0].message.content
                p.write_text(json.dumps({"description": out, "model": self.model}, ensure_ascii=False, indent=2))
                return out
        except TimeoutException as e:
            print(f"    [TIMEOUT] GPT describe_image: {e}")
            return ""
        except Exception as e:
            print(f"    [ERROR] GPT describe_image failed: {type(e).__name__}: {e}")
            return ""

    def grade_answer(self, cache_key: str, question: str, choices: Any,
                     ground_truth: str, model_answer: str, timeout_seconds: int = 120) -> Dict[str, Any]:
        """Grade model answer against ground truth using GPT."""
        p = self.grade_dir / f"{self._safe_key(cache_key)}.json"
        if p.exists():
            return json.loads(p.read_text())

        grading_prompt = f"""Question: {question}

Choices: {json.dumps(choices) if choices else "N/A (free-form)"}

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

Is the model's answer correct? Respond with JSON:
{{"is_correct": true/false, "extracted_answer": "...", "reason": "brief explanation"}}"""

        try:
            with timeout_context(timeout_seconds, f"GPT grade_answer timed out after {timeout_seconds}s"):
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.GRADING_SYSTEM},
                        {"role": "user", "content": grading_prompt},
                    ],
                    max_completion_tokens=1000,
                    timeout=60.0,
                )
                raw = resp.choices[0].message.content

                raw_clean = raw.strip()
                if raw_clean.startswith("```"):
                    lines = raw_clean.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw_clean = "\n".join(lines)

                try:
                    data = json.loads(raw_clean)
                except json.JSONDecodeError:
                    data = {"is_correct": "true" in raw.lower()[:100], "raw": raw}

                data["grader_model"] = self.model
                p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                return data

        except TimeoutException as e:
            print(f"    [TIMEOUT] GPT grade_answer: {e}")
            return {"is_correct": False, "error": "timeout"}
        except Exception as e:
            print(f"    [ERROR] GPT grade_answer failed: {type(e).__name__}: {e}")
            return {"is_correct": False, "error": str(e)}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Model paths (required for resolving aliases)
    sft_model: str = "OpenMMReasoner/OpenMMReasoner-ColdStart"
    rl_model: str = "OpenMMReasoner/OpenMMReasoner-RL"
    base_model: Optional[str] = None

    # Sampling
    sample_seed: int = 123
    sample_size: int = 100

    # Inference settings
    max_new_tokens: int = 2048
    temperature: float = 0.0
    max_image_pixels: int = 1280 * 720
    inference_timeout: int = 1200  # 20 minutes default

    # Output
    output_dir: str = "./runs"
    run_name: Optional[str] = None

    # Models to evaluate
    models: Optional[List[str]] = None

    # GPT settings
    use_gpt_descriptions: bool = True
    use_gpt_grader: bool = True
    openai_model: str = "gpt-4o-mini"

    # Reuse results from previous run
    reuse_from: Optional[str] = None

    # Experiment type: 'v1' (general) or 'v2' (finer splits)
    experiment_type: str = "v2"


# ============================================================================
# System Prompts
# ============================================================================

# For REASONING models: ColdStart, RL, MMR1, and their splices
SYSTEM_PROMPT_REASONER = """You are a helpful assistant. When the user asks a question, your response must include two parts: first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."""

# For REVISUAL-R1 models
SYSTEM_PROMPT_REVISUAL = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."""

# For DEFAULT models: Qwen VL and layer-replaced variants
SYSTEM_PROMPT_DEFAULT = None

QUERY_PROMPT_PREFIX = 'Please solve the problem step by step and put your answer in one "\\boxed{}".\n\n'


# ============================================================================
# Dataset Loading
# ============================================================================

class BaseDataset:
    """Base class for evaluation datasets."""

    def __init__(self, dataset_name: str, split: str, cache_path: str):
        self.dataset_name = dataset_name
        self.split = split
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.descriptions = {}
        self.dataset = None

    def get_item(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)


class MathVistaDataset(BaseDataset):
    """Load MathVista dataset."""

    def __init__(self, split: str = "testmini", cache_path: str = "./cache/mathvista_desc.json"):
        super().__init__("AI4Math/MathVista", split, cache_path)
        print(f"Loading AI4Math/MathVista ({split})...")
        self.dataset = load_dataset("AI4Math/MathVista", split=split)
        print(f"  Loaded {len(self.dataset)} samples")

    def get_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image_id = str(item.get("pid", idx))
        image = item.get("decoded_image") or item.get("image")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image is not None and hasattr(image, 'convert'):
            image = image.convert("RGB")

        question = item.get("question", "")
        choices = item.get("choices", None)
        question_type = item.get("question_type", "free_form")

        if question_type == "multi_choice" and choices:
            choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
            formatted_choices = []
            for i, choice in enumerate(choices):
                if i < len(choice_labels):
                    formatted_choices.append(f"{choice_labels[i]}. {choice}")
            question = question + "\n\nChoices:\n" + "\n".join(formatted_choices)

        metadata = item.get("metadata", {}) or {}
        return {
            "id": image_id,
            "idx": idx,
            "image": image,
            "question": question,
            "answer": str(item.get("answer", "")),
            "answer_type": item.get("answer_type", "text"),
            "question_type": question_type,
            "choices": choices,
            "metadata": metadata,
            "context": metadata.get("context", "unknown"),
            "task": metadata.get("task", "unknown"),
            "category": metadata.get("category", "unknown"),
            "description": "",
        }


class MATH500Dataset(BaseDataset):
    """Load MATH-500 dataset (text-only)."""

    def __init__(self, split: str = "test", cache_path: str = "./cache/math500_desc.json"):
        super().__init__("HuggingFaceH4/MATH-500", split, cache_path)
        print(f"Loading HuggingFaceH4/MATH-500 ({split})...")
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
        print(f"  Loaded {len(self.dataset)} samples")

    def get_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        return {
            "id": str(item.get("unique_id", idx)),
            "idx": idx,
            "image": None,
            "question": item.get("problem", ""),
            "answer": str(item.get("answer", "")),
            "answer_type": "text",
            "question_type": "free_form",
            "choices": None,
            "metadata": {"level": item.get("level"), "subject": item.get("subject")},
            "level": item.get("level", 1),
            "subject": item.get("subject", "unknown"),
            "description": "",
        }


def subsample_indices(n: int, k: int, seed: int) -> List[int]:
    """Random subsample k indices from n with fixed seed."""
    rng = np.random.RandomState(seed)
    if k >= n:
        return list(range(n))
    return sorted(rng.choice(np.arange(n), size=k, replace=False).tolist())


def ensure_split_file(split_path: Path, all_ids: List[str], k: int, seed: int) -> List[str]:
    """Persist a fixed subsample of ids and return them."""
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        return [json.loads(l)["id"] for l in split_path.read_text().splitlines() if l.strip()]
    idxs = subsample_indices(len(all_ids), k, seed)
    chosen = [all_ids[i] for i in idxs]
    with open(split_path, "w") as f:
        for _id in chosen:
            f.write(json.dumps({"id": _id}) + "\n")
    return chosen


# ============================================================================
# GPT Evaluator (OpenAI API)
# ============================================================================

class GPTEvaluator:
    """OpenAI GPT-based VLM evaluation."""

    SYSTEM_PROMPT = """You are a helpful assistant that solves math and reasoning problems.
When answering, first think through the problem step by step, then provide your final answer.
Put your final answer in \\boxed{} format."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        max_image_pixels: int = 1280 * 720,
        timeout: int = 120,
    ):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed. Install with: pip install openai")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_image_pixels = max_image_pixels
        self.timeout = timeout
        self._client = OpenAI()

    @staticmethod
    def _image_to_data_url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _is_responses_api_model(self) -> bool:
        """Check if this model uses the new responses API (gpt-5)."""
        return "gpt-5" in self.model.lower()

    def _is_reasoning_model(self) -> bool:
        """Check if this is an OpenAI reasoning model (o1/o3/o4)."""
        model_lower = self.model.lower()
        return any(p in model_lower for p in ["o1", "o3", "o4"])

    def _build_messages_chat(self, user_content, include_system: bool = True):
        """Build messages list for chat.completions API."""
        if self._is_reasoning_model():
            if isinstance(user_content, str):
                combined_text = f"{self.SYSTEM_PROMPT}\n\n{user_content}"
                return [{"role": "user", "content": combined_text}]
            else:
                combined_content = [{"type": "text", "text": self.SYSTEM_PROMPT + "\n\n"}] + user_content
                return [{"role": "user", "content": combined_content}]
        else:
            messages = []
            if include_system:
                messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
            messages.append({"role": "user", "content": user_content})
            return messages

    def _call_chat_api(self, messages):
        """Call the chat.completions API."""
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_new_tokens,
            "timeout": 60.0,
        }

        if not self._is_reasoning_model():
            api_kwargs["temperature"] = self.temperature

        resp = self._client.chat.completions.create(**api_kwargs)
        return resp.choices[0].message.content or ""

    def evaluate_single(self, item: Dict, input_type: str) -> Optional[Dict]:
        """Evaluate a single item using OpenAI API."""
        start_time = time.time()

        if input_type == "image_question":
            if item["image"] is None:
                return None
            resized_image = resize_image_if_needed(item["image"], self.max_image_pixels)
            data_url = self._image_to_data_url(resized_image)
            question_text = QUERY_PROMPT_PREFIX + item["question"]
            user_content = [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": question_text},
            ]

        elif input_type == "black_image_question":
            if item.get("image") is None:
                return None
            black = make_black_image(item["image"])
            resized_image = resize_image_if_needed(black, self.max_image_pixels)
            data_url = self._image_to_data_url(resized_image)
            question_text = QUERY_PROMPT_PREFIX + item["question"]
            user_content = [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": question_text},
            ]

        elif input_type == "black_image_description_question":
            if item.get("image") is None or not item.get("description"):
                return None
            black = make_black_image(item["image"])
            resized_image = resize_image_if_needed(black, self.max_image_pixels)
            data_url = self._image_to_data_url(resized_image)
            text = f"""{QUERY_PROMPT_PREFIX}Image Description:
{item['description']}

Question:
{item['question']}"""
            user_content = [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": text},
            ]

        elif input_type == "question_only":
            user_content = QUERY_PROMPT_PREFIX + item["question"]

        else:
            return None

        messages = self._build_messages_chat(user_content)
        response = self._call_chat_api(messages)

        elapsed = time.time() - start_time

        predicted = self._extract_answer(response, item.get("question_type", "free_form"))
        correct = self._check_answer(
            predicted,
            item["answer"],
            item.get("answer_type", "text"),
            item.get("question_type", "free_form"),
            item.get("choices"),
            response
        )

        return {
            "id": item["id"],
            "input_type": input_type,
            "question": item["question"],
            "gold_answer": item["answer"],
            "predicted": predicted,
            "correct": correct,
            "full_response": response,
            "model_type": "gpt",
            "inference_time": round(elapsed, 2),
        }

    def _extract_answer(self, response: str, question_type: str = "free_form") -> str:
        if not response:
            return ""

        extracted = extract_answer_tag(response)
        if extracted:
            extracted_clean = extracted.strip()
            m = re.match(r'^\s*([A-Ha-h0-9])\s*[\.\)\:\-]\s*(.+?)\s*$', extracted_clean)
            if m:
                return m.group(2).strip()
            if len(extracted_clean) == 1 and extracted_clean.upper() in "ABCDEFGH":
                return extracted_clean.upper()
            return extracted_clean

        if question_type == "multi_choice":
            mcq_answer = parse_mcq(response)
            if mcq_answer:
                return mcq_answer

        boxed = extract_boxed_answer(response)
        if boxed:
            return boxed

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

        numbers = re.findall(r'(-?\d+\.?\d*)', response)
        if numbers:
            return numbers[-1]

        return response.strip()[-50:] if response else ""

    def _check_answer(self, predicted: str, gold: str, answer_type: str,
                      question_type: str, choices: Optional[List[str]],
                      full_response: str) -> bool:
        if not predicted or not gold:
            return False

        pred = predicted.strip()
        gold_clean = gold.strip()
        pred_norm = pred.lower().replace(" ", "")
        gold_norm = gold_clean.lower().replace(" ", "")

        if len(gold_clean) == 1 and gold_clean.upper() in "ABCDEFGH":
            pred_letter = parse_mcq(pred)
            if pred_letter and pred_letter == gold_clean.upper():
                return True
            if full_response:
                pred_letter = parse_mcq(full_response)
                if pred_letter and pred_letter == gold_clean.upper():
                    return True
            return False

        if question_type == "multi_choice" and choices:
            if len(pred.strip()) == 1 and pred.strip().upper() in "ABCDEFGH":
                idx = ord(pred.strip().upper()) - ord("A")
                if 0 <= idx < len(choices):
                    mapped = str(choices[idx]).strip().lower().replace(" ", "")
                    if mapped == gold_norm:
                        return True

        if pred_norm == gold_norm:
            return True

        if gold_norm in pred_norm and len(gold_norm) >= 0.9 * len(pred_norm):
            return True
        if pred_norm in gold_norm and len(pred_norm) >= 0.9 * len(gold_norm):
            return True

        if MATH_VERIFY_AVAILABLE:
            try:
                if verify(parse(pred), parse(gold_clean)):
                    return True
            except:
                pass

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
# Claude Evaluator (Anthropic API)
# ============================================================================

class ClaudeEvaluator:
    """Anthropic Claude-based VLM evaluation."""

    SYSTEM_PROMPT = """You are a helpful assistant that solves math and reasoning problems.
When answering, first think through the problem step by step, then provide your final answer.
Put your final answer in \\boxed{} format."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        max_image_pixels: int = 1280 * 720,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed. Install with: pip install anthropic")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_image_pixels = max_image_pixels
        self._client = anthropic.Anthropic()

    @staticmethod
    def _image_to_base64(img: Image.Image) -> tuple:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return ("image/png", b64)

    def evaluate_single(self, item: Dict, input_type: str) -> Optional[Dict]:
        """Evaluate a single item using Anthropic API."""
        start_time = time.time()

        if input_type == "image_question":
            if item["image"] is None:
                return None
            resized_image = resize_image_if_needed(item["image"], self.max_image_pixels)
            media_type, b64_data = self._image_to_base64(resized_image)
            question_text = QUERY_PROMPT_PREFIX + item["question"]
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    }
                },
                {"type": "text", "text": question_text},
            ]

        elif input_type == "black_image_question":
            if item.get("image") is None:
                return None
            black = make_black_image(item["image"])
            resized_image = resize_image_if_needed(black, self.max_image_pixels)
            media_type, b64_data = self._image_to_base64(resized_image)
            question_text = QUERY_PROMPT_PREFIX + item["question"]
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    }
                },
                {"type": "text", "text": question_text},
            ]

        elif input_type == "black_image_description_question":
            if item.get("image") is None or not item.get("description"):
                return None
            black = make_black_image(item["image"])
            resized_image = resize_image_if_needed(black, self.max_image_pixels)
            media_type, b64_data = self._image_to_base64(resized_image)
            text = f"""{QUERY_PROMPT_PREFIX}Image Description:
{item['description']}

Question:
{item['question']}"""
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    }
                },
                {"type": "text", "text": text},
            ]

        elif input_type == "question_only":
            question_text = QUERY_PROMPT_PREFIX + item["question"]
            content = [{"type": "text", "text": question_text}]

        else:
            return None

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        response = message.content[0].text if message.content else ""
        elapsed = time.time() - start_time

        predicted = self._extract_answer(response, item.get("question_type", "free_form"))
        correct = self._check_answer(
            predicted,
            item["answer"],
            item.get("answer_type", "text"),
            item.get("question_type", "free_form"),
            item.get("choices"),
            response
        )

        return {
            "id": item["id"],
            "input_type": input_type,
            "question": item["question"],
            "gold_answer": item["answer"],
            "predicted": predicted,
            "correct": correct,
            "full_response": response,
            "model_type": "claude",
            "inference_time": round(elapsed, 2),
        }

    def _extract_answer(self, response: str, question_type: str = "free_form") -> str:
        if not response:
            return ""

        extracted = extract_answer_tag(response)
        if extracted:
            extracted_clean = extracted.strip()
            m = re.match(r'^\s*([A-Ha-h0-9])\s*[\.\)\:\-]\s*(.+?)\s*$', extracted_clean)
            if m:
                return m.group(2).strip()
            if len(extracted_clean) == 1 and extracted_clean.upper() in "ABCDEFGH":
                return extracted_clean.upper()
            return extracted_clean

        if question_type == "multi_choice":
            mcq_answer = parse_mcq(response)
            if mcq_answer:
                return mcq_answer

        boxed = extract_boxed_answer(response)
        if boxed:
            return boxed

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

        numbers = re.findall(r'(-?\d+\.?\d*)', response)
        if numbers:
            return numbers[-1]

        return response.strip()[-50:] if response else ""

    def _check_answer(self, predicted: str, gold: str, answer_type: str,
                      question_type: str, choices: Optional[List[str]],
                      full_response: str) -> bool:
        if not predicted or not gold:
            return False

        pred = predicted.strip()
        gold_clean = gold.strip()
        pred_norm = pred.lower().replace(" ", "")
        gold_norm = gold_clean.lower().replace(" ", "")

        if len(gold_clean) == 1 and gold_clean.upper() in "ABCDEFGH":
            pred_letter = parse_mcq(pred)
            if pred_letter and pred_letter == gold_clean.upper():
                return True
            if full_response:
                pred_letter = parse_mcq(full_response)
                if pred_letter and pred_letter == gold_clean.upper():
                    return True
            return False

        if question_type == "multi_choice" and choices:
            if len(pred.strip()) == 1 and pred.strip().upper() in "ABCDEFGH":
                idx = ord(pred.strip().upper()) - ord("A")
                if 0 <= idx < len(choices):
                    mapped = str(choices[idx]).strip().lower().replace(" ", "")
                    if mapped == gold_norm:
                        return True

        if pred_norm == gold_norm:
            return True

        if gold_norm in pred_norm and len(gold_norm) >= 0.9 * len(pred_norm):
            return True
        if pred_norm in gold_norm and len(pred_norm) >= 0.9 * len(gold_norm):
            return True

        if MATH_VERIFY_AVAILABLE:
            try:
                if verify(parse(pred), parse(gold_clean)):
                    return True
            except:
                pass

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
# HuggingFace Evaluator
# ============================================================================

class HFEvaluator:
    """HuggingFace-based VLM evaluation."""

    def __init__(
        self,
        model,
        processor,
        model_type: str = "reasoner",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        max_image_pixels: int = 1280 * 720,
        inference_timeout: int = 1200
    ):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_image_pixels = max_image_pixels
        self.inference_timeout = inference_timeout

    def _get_system_prompt(self) -> Optional[str]:
        """Get system prompt based on model type."""
        if self.model_type == "revisual":
            return SYSTEM_PROMPT_REVISUAL
        elif self.model_type == "reasoner":
            return SYSTEM_PROMPT_REASONER
        return SYSTEM_PROMPT_DEFAULT

    def evaluate_single(self, item: Dict, input_type: str) -> Optional[Dict]:
        """Evaluate a single item."""
        system_prompt = self._get_system_prompt()
        start_time = time.time()

        if input_type == "image_question":
            if item["image"] is None:
                return None
            resized_image = resize_image_if_needed(item["image"], self.max_image_pixels)
            question_text = QUERY_PROMPT_PREFIX + item["question"]
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": question_text}
                ]
            })
            image = resized_image

        elif input_type == "black_image_question":
            if item.get("image") is None:
                return None
            black = make_black_image(item["image"])
            resized_image = resize_image_if_needed(black, self.max_image_pixels)
            question_text = QUERY_PROMPT_PREFIX + item["question"]
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": question_text}
                ]
            })
            image = resized_image

        elif input_type == "black_image_description_question":
            if item.get("image") is None or not item.get("description"):
                return None
            black = make_black_image(item["image"])
            resized_image = resize_image_if_needed(black, self.max_image_pixels)
            text = f"""{QUERY_PROMPT_PREFIX}Image Description:
{item['description']}

Question:
{item['question']}"""
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": text}
                ]
            })
            image = resized_image

        elif input_type == "question_only":
            question_text = QUERY_PROMPT_PREFIX + item["question"]
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question_text})
            image = None

        else:
            return None

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if image is not None:
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
            )

        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        try:
            with timeout_context(self.inference_timeout, f"Inference timed out after {self.inference_timeout} seconds"):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature if self.temperature > 0 else None,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                    )
        except TimeoutException as e:
            print(f"    [TIMEOUT] Sample {item.get('id', 'unknown')}: {e}")
            del inputs
            torch.cuda.empty_cache()
            return {
                "id": item["id"],
                "input_type": input_type,
                "question": item["question"],
                "gold_answer": item["answer"],
                "predicted": "",
                "correct": False,
                "full_response": f"[TIMEOUT after {self.inference_timeout}s]",
                "model_type": self.model_type,
                "timed_out": True,
                "inference_time": time.time() - start_time,
            }

        elapsed = time.time() - start_time
        input_len = inputs["input_ids"].shape[1]
        response = self.processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        predicted = self._extract_answer(response, item.get("question_type", "free_form"))
        correct = self._check_answer(
            predicted,
            item["answer"],
            item.get("answer_type", "text"),
            item.get("question_type", "free_form"),
            item.get("choices"),
            response
        )

        result = {
            "id": item["id"],
            "input_type": input_type,
            "question": item["question"],
            "gold_answer": item["answer"],
            "predicted": predicted,
            "correct": correct,
            "full_response": response,
            "model_type": self.model_type,
            "timed_out": False,
            "inference_time": round(elapsed, 2),
        }

        del inputs, outputs
        torch.cuda.empty_cache()

        return result

    def _extract_answer(self, response: str, question_type: str = "free_form") -> str:
        if not response:
            return ""
        extracted = extract_answer_tag(response)
        if extracted:
            extracted_clean = extracted.strip()
            m = re.match(r'^\s*([A-Ha-h0-9])\s*[\.\)\:\-]\s*(.+?)\s*$', extracted_clean)
            if m:
                return m.group(2).strip()
            if len(extracted_clean) == 1 and extracted_clean.upper() in "ABCDEFGH":
                return extracted_clean.upper()
            return extracted_clean
        if question_type == "multi_choice":
            mcq_answer = parse_mcq(response)
            if mcq_answer:
                return mcq_answer
        boxed = extract_boxed_answer(response)
        if boxed:
            return boxed
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
        numbers = re.findall(r'(-?\d+\.?\d*)', response)
        if numbers:
            return numbers[-1]
        return response.strip()[-50:] if response else ""

    def _check_answer(self, predicted: str, gold: str, answer_type: str,
                      question_type: str, choices: Optional[List[str]],
                      full_response: str) -> bool:
        if not predicted or not gold:
            return False
        pred = predicted.strip()
        gold_clean = gold.strip()
        pred_norm = pred.lower().replace(" ", "")
        gold_norm = gold_clean.lower().replace(" ", "")

        if len(gold_clean) == 1 and gold_clean.upper() in "ABCDEFGH":
            pred_letter = parse_mcq(pred)
            if pred_letter and pred_letter == gold_clean.upper():
                return True
            if full_response:
                pred_letter = parse_mcq(full_response)
                if pred_letter and pred_letter == gold_clean.upper():
                    return True
            return False

        if question_type == "multi_choice" and choices:
            if len(pred.strip()) == 1 and pred.strip().upper() in "ABCDEFGH":
                idx = ord(pred.strip().upper()) - ord("A")
                if 0 <= idx < len(choices):
                    mapped = str(choices[idx]).strip().lower().replace(" ", "")
                    if mapped == gold_norm:
                        return True

        if pred_norm == gold_norm:
            return True

        if gold_norm in pred_norm and len(gold_norm) >= 0.9 * len(pred_norm):
            return True
        if pred_norm in gold_norm and len(pred_norm) >= 0.9 * len(gold_norm):
            return True

        if MATH_VERIFY_AVAILABLE:
            try:
                if verify(parse(pred), parse(gold_clean)):
                    return True
            except:
                pass

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
# Model Manager
# ============================================================================

class ModelManager:
    """Manage model loading."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.processor = None

    def _load_model(self, model_path: str):
        """Load a model from HuggingFace."""
        print(f"  Loading model: {model_path}")
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        ).eval()

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        return model

    def _resolve_model_path(self, name: str) -> str:
        """Resolve model aliases to paths."""
        name_lower = name.lower().strip()
        if name_lower in ("sft", "coldstart"):
            return self.config.sft_model
        if name_lower in ("rl", "final"):
            return self.config.rl_model
        if name_lower == "base":
            return self.config.base_model
        return name

    def _get_layers(self, model):
        """Get transformer layers from model."""
        if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            return model.model.language_model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
            return model.model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Cannot find layers")

    def get_model(self, model_name: str) -> Tuple[Any, str]:
        """Load a model by name. Returns (model, model_type)."""

        model_type = detect_model_type(model_name)
        prompt_style = {
            'gpt': 'OpenAI API',
            'claude': 'Anthropic API',
            'revisual': 'Revisual-R1 (<think> + \\boxed{})',
            'reasoner': 'reasoning (<think> + <answer>)',
            'base': 'default (no special prompting)'
        }.get(model_type, 'default')
        print(f"  Model type: {model_type} ({prompt_style})")

        # GPT models: return model name directly
        if model_type == "gpt":
            print(f"  Using OpenAI API for model: {model_name}")
            return model_name, model_type

        # Claude models: return model name directly
        if model_type == "claude":
            print(f"  Using Anthropic API for model: {model_name}")
            return model_name, model_type

        # THREE-WAY splice: "A early 1/3 + B middle 1/3 + C late 1/3 layers"
        three_way_splice_match = re.search(
            r"^(?P<early>.+?)\s+early\s+1/3\s*\+\s*(?P<middle>.+?)\s+middle\s+1/3\s*\+\s*(?P<late>.+?)\s+late\s+1/3\s+layers?\s*$",
            model_name, re.IGNORECASE
        )
        if three_way_splice_match:
            early_name = three_way_splice_match.group("early").strip()
            middle_name = three_way_splice_match.group("middle").strip()
            late_name = three_way_splice_match.group("late").strip()

            early_path = self._resolve_model_path(early_name)
            middle_path = self._resolve_model_path(middle_name)
            late_path = self._resolve_model_path(late_name)

            print(f"  Creating three-way splice: early 1/3 from {early_path}, middle 1/3 from {middle_path}, late 1/3 from {late_path}")
            from transformers import Qwen2_5_VLForConditionalGeneration

            early_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                early_path, torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()
            middle_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                middle_path, torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()
            late_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                late_path, torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()

            early_layers = self._get_layers(early_model)
            middle_layers = self._get_layers(middle_model)
            late_layers = self._get_layers(late_model)

            n_layers = len(early_layers)
            cut1 = n_layers // 3
            cut2 = 2 * n_layers // 3

            print(f"    Total layers: {n_layers}, cut1: {cut1}, cut2: {cut2}")

            with torch.no_grad():
                for i in range(cut1, cut2):
                    early_layers[i].load_state_dict(middle_layers[i].state_dict())
                for i in range(cut2, n_layers):
                    early_layers[i].load_state_dict(late_layers[i].state_dict())

            del middle_model, late_model
            gc.collect()
            early_model = early_model.to("cuda")
            torch.cuda.empty_cache()

            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(early_path, trust_remote_code=True)
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            return early_model, model_type

        # TWO-WAY splice: "X early N/3 layers + Y late M/3 layers"
        splice_match = re.search(
            r"^(?P<early>.+?)\s+early\s+(?P<early_frac>[12])/3\s+layers?\s*\+\s*(?P<late>.+?)\s+late\s+(?P<late_frac>[12])/3\s+layers?\s*$",
            model_name, re.IGNORECASE
        )
        if splice_match:
            early_name = splice_match.group("early").strip()
            late_name = splice_match.group("late").strip()
            early_frac = int(splice_match.group("early_frac"))
            late_frac = int(splice_match.group("late_frac"))

            if early_frac + late_frac != 3:
                raise ValueError(f"Splice fractions must add up to 3/3, got {early_frac}/3 + {late_frac}/3")

            early_path = self._resolve_model_path(early_name)
            late_path = self._resolve_model_path(late_name)

            print(f"  Creating splice: early {early_frac}/3 from {early_path}, late {late_frac}/3 from {late_path}")
            from transformers import Qwen2_5_VLForConditionalGeneration

            early_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                early_path, torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()
            late_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                late_path, torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()

            early_layers = self._get_layers(early_model)
            late_layers = self._get_layers(late_model)
            cut = early_frac * len(early_layers) // 3

            with torch.no_grad():
                for i in range(cut, len(early_layers)):
                    early_layers[i].load_state_dict(late_layers[i].state_dict())

            del late_model
            gc.collect()
            early_model = early_model.to("cuda")
            torch.cuda.empty_cache()

            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(early_path, trust_remote_code=True)
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            return early_model, model_type

        # Layer ablation: "base layers 0-1 ablate noise=0.5"
        ablate_match = re.search(
            r"^(?P<base>.+?)\s+layers?\s+(?P<start>\d+)-(?P<end>\d+)\s+ablate\s+noise=(?P<noise>[\d.]+)\s*$",
            model_name, re.IGNORECASE
        )
        if ablate_match:
            base_path = ablate_match.group("base").strip()
            start = int(ablate_match.group("start"))
            end = int(ablate_match.group("end")) + 1
            noise_ratio = float(ablate_match.group("noise"))

            print(f"  Creating layer ablation: base={base_path}, layers={start}-{end-1}, noise={noise_ratio}")
            from transformers import Qwen2_5_VLForConditionalGeneration

            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_path, torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()

            base_layers = self._get_layers(base_model)

            with torch.no_grad():
                for i in range(start, end):
                    base_state = base_layers[i].state_dict()
                    for key in base_state:
                        weight = base_state[key]
                        std = weight.std().item()
                        random_weight = torch.randn_like(weight) * std
                        base_state[key] = (1 - noise_ratio) * weight + noise_ratio * random_weight
                    base_layers[i].load_state_dict(base_state)

            base_model = base_model.to("cuda")
            torch.cuda.empty_cache()

            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            return base_model, model_type

        # Layer range interpolation: "base layers 0-1 interpolate donor alpha=0.5 [component=mlp|attn|all]"
        layer_range_match = re.search(
            r"^(?P<base>.+?)\s+layers?\s+(?P<start>\d+)-(?P<end>\d+)\s+interpolate\s+(?P<donor>.+?)\s+alpha=(?P<alpha>[\d.]+)(?:\s+component=(?P<component>mlp|attn|all))?\s*$",
            model_name, re.IGNORECASE
        )
        if layer_range_match:
            base_path = layer_range_match.group("base").strip()
            start = int(layer_range_match.group("start"))
            end = int(layer_range_match.group("end")) + 1
            donor_path = layer_range_match.group("donor").strip()
            alpha = float(layer_range_match.group("alpha"))
            component = (layer_range_match.group("component") or "all").lower()

            print(f"  Creating layer range interpolation: base={base_path}, layers={start}-{end-1}, donor={donor_path}, alpha={alpha}, component={component}")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM

            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_path, torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()
            donor_model = AutoModelForCausalLM.from_pretrained(
                donor_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
            ).eval()

            base_layers = self._get_layers(base_model)
            donor_layers = self._get_layers(donor_model)

            with torch.no_grad():
                for i in range(start, end):
                    base_state = base_layers[i].state_dict()
                    donor_state = donor_layers[i].state_dict()
                    for key in base_state:
                        if key in donor_state:
                            should_interpolate = False
                            if component == "all":
                                should_interpolate = True
                            elif component == "mlp":
                                should_interpolate = key.startswith("mlp.")
                            elif component == "attn":
                                should_interpolate = key.startswith("self_attn.")

                            if should_interpolate:
                                base_state[key] = (1 - alpha) * base_state[key] + alpha * donor_state[key]
                    base_layers[i].load_state_dict(base_state)

            del donor_model
            gc.collect()
            base_model = base_model.to("cuda")
            torch.cuda.empty_cache()

            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            return base_model, model_type

        # Segment interpolation: "base early/middle/late 1/3 layers interpolate donor alpha=0.01"
        interp_match = re.search(
            r"^(?P<base>.+?)\s+(?P<seg>early|middle|late)\s+1/3\s+layers?\s+interpolate\s+(?P<donor>.+?)\s+alpha=(?P<alpha>[\d.]+)\s*$",
            model_name, re.IGNORECASE
        )
        replace_match = re.search(
            r"^(?P<base>.+?)\s+(?P<seg>early|middle|late)\s+1/3\s+layers?\s+replaced\s+by\s+(?P<donor>.+?)\s*$",
            model_name, re.IGNORECASE
        )

        match = interp_match or replace_match
        if match:
            base_path = match.group("base").strip()
            segment = match.group("seg").lower()
            donor_path = match.group("donor").strip()
            alpha = float(interp_match.group("alpha")) if interp_match else 1.0

            print(f"  Creating interpolation: base={base_path}, segment={segment}, donor={donor_path}, alpha={alpha}")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM

            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_path, torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()
            donor_model = AutoModelForCausalLM.from_pretrained(
                donor_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
            ).eval()

            base_layers = self._get_layers(base_model)
            donor_layers = self._get_layers(donor_model)
            n = len(base_layers)
            third = n // 3

            if segment == "early":
                start, end = 0, third
            elif segment == "middle":
                start, end = third, 2 * third
            else:
                start, end = 2 * third, n

            with torch.no_grad():
                for i in range(start, end):
                    base_state = base_layers[i].state_dict()
                    donor_state = donor_layers[i].state_dict()
                    for key in base_state:
                        if key in donor_state:
                            base_state[key] = (1 - alpha) * base_state[key] + alpha * donor_state[key]
                    base_layers[i].load_state_dict(base_state)

            del donor_model
            gc.collect()
            base_model = base_model.to("cuda")
            torch.cuda.empty_cache()

            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            return base_model, model_type

        # Simple model path
        path = self._resolve_model_path(model_name)
        model = self._load_model(path)
        return model, model_type

    def cleanup(self, model):
        """Cleanup model."""
        del model
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# Transition Statistics
# ============================================================================

def compute_transition_stats(labels_a: Dict[str, bool], labels_b: Dict[str, bool]) -> Dict[str, Any]:
    """Compute transition statistics between condition A and B."""
    ids = sorted(set(labels_a.keys()) & set(labels_b.keys()))
    n_tt = n_tf = n_ft = n_ff = 0

    for _id in ids:
        a = labels_a[_id]
        b = labels_b[_id]
        if a and b:
            n_tt += 1
        elif a and not b:
            n_tf += 1
        elif not a and b:
            n_ft += 1
        else:
            n_ff += 1

    N = len(ids)
    acc_a = (n_tt + n_tf) / N if N else 0.0
    acc_b = (n_tt + n_ft) / N if N else 0.0

    return {
        "N": N,
        "acc_A": round(acc_a, 4),
        "acc_B": round(acc_b, 4),
        "delta_acc": round(acc_a - acc_b, 4),
        "n_TT": n_tt,
        "n_TF": n_tf,
        "n_FT": n_ft,
        "n_FF": n_ff,
        "net_gain": n_ft - n_tf,
        "side_effect_rate": round(n_tf / (n_tt + n_tf), 4) if (n_tt + n_tf) > 0 else 0.0,
        "lift_rate": round(n_ft / (n_ff + n_ft), 4) if (n_ff + n_ft) > 0 else 0.0,
    }


# ============================================================================
# Experiment Setup
# ============================================================================

# Contexts for pure perception (v2 finer splits)
PERCEPTION_CONTEXTS = [
    "natural image", "bar chart", "table", "pie chart", "scatter plot", "line plot",
]

# Contexts for geometry (v2 finer splits)
GEOMETRY_CONTEXTS = ["geometry diagram"]


def get_experiments_v1(mv, m500, config, run_dir):
    """Get experiment structure for v1 (general/math VQA splits)."""
    gen_idxs = [i for i in range(len(mv)) if (mv.dataset[i].get("metadata") or {}).get("category") == "general-vqa"]
    mt_idxs = [i for i in range(len(mv)) if (mv.dataset[i].get("metadata") or {}).get("category") == "math-targeted-vqa"]

    print(f"  General VQA samples: {len(gen_idxs)}")
    print(f"  Math VQA samples: {len(mt_idxs)}")
    print(f"  MATH-500 samples: {len(m500)}")

    def build_id_map(ds, idxs):
        ids, mp = [], {}
        for ix in idxs:
            item = ds.get_item(ix)
            sid = str(item["id"])
            ids.append(sid)
            mp[sid] = ix
        return ids, mp

    gen_ids, gen_mp = build_id_map(mv, gen_idxs)
    mt_ids, mt_mp = build_id_map(mv, mt_idxs)
    m500_ids = [str(m500.get_item(i)["id"]) for i in range(len(m500))]
    m500_mp = {sid: i for i, sid in enumerate(m500_ids)}

    chosen_gen = ensure_split_file(
        run_dir / "splits" / f"general_vqa_seed{config.sample_seed}_n{config.sample_size}.jsonl",
        gen_ids, config.sample_size, config.sample_seed
    )
    chosen_mt = ensure_split_file(
        run_dir / "splits" / f"math_vqa_seed{config.sample_seed}_n{config.sample_size}.jsonl",
        mt_ids, config.sample_size, config.sample_seed
    )
    chosen_m500 = ensure_split_file(
        run_dir / "splits" / f"math500_seed{config.sample_seed}_n{config.sample_size}.jsonl",
        m500_ids, config.sample_size, config.sample_seed
    )

    print(f"\n  Sampled: {len(chosen_gen)} General VQA, {len(chosen_mt)} Math VQA, {len(chosen_m500)} MATH-500")

    return {
        "perception_general_vqa": {
            "dataset": mv,
            "indices": [gen_mp[s] for s in chosen_gen if s in gen_mp],
            "settings": [
                ("A_image_prompt", "image_question"),
                ("B_black_image_prompt", "black_image_question"),
            ],
            "type": "perception",
            "description": "General VQA (perception test)",
        },
        "alignment_math_vqa": {
            "dataset": mv,
            "indices": [mt_mp[s] for s in chosen_mt if s in mt_mp],
            "settings": [
                ("A_image_prompt", "image_question"),
                ("B_black_desc_prompt", "black_image_description_question"),
            ],
            "type": "alignment",
            "description": "Math VQA (alignment test)",
        },
        "reasoning_math500": {
            "dataset": m500,
            "indices": [m500_mp[s] for s in chosen_m500 if s in m500_mp],
            "settings": [
                ("A_prompt_only", "question_only"),
            ],
            "type": "reasoning",
            "description": "MATH-500 (reasoning test)",
        },
    }


def get_experiments_v2(mv, m500, config, run_dir):
    """Get experiment structure for v2 (finer-grained splits)."""
    perception_idxs = []
    for i in range(len(mv)):
        meta = mv.dataset[i].get("metadata") or {}
        context = meta.get("context", "")
        if context in PERCEPTION_CONTEXTS:
            perception_idxs.append(i)

    geometry_idxs = []
    for i in range(len(mv)):
        meta = mv.dataset[i].get("metadata") or {}
        context = meta.get("context", "")
        task = meta.get("task", "")
        if context in GEOMETRY_CONTEXTS or task == "geometry problem solving":
            geometry_idxs.append(i)

    hard_math_idxs = []
    for i in range(len(m500)):
        item = m500.dataset[i]
        level = item.get("level", 1)
        if level >= 4:
            hard_math_idxs.append(i)

    print(f"  Finer-grained splits:")
    print(f"    Pure Perception (charts, tables, natural images): {len(perception_idxs)}")
    print(f"    Vision+Reasoning (geometry): {len(geometry_idxs)}")
    print(f"    Pure Reasoning (MATH-500 L4-5): {len(hard_math_idxs)}")

    def build_id_map(ds, idxs):
        ids, mp = [], {}
        for ix in idxs:
            item = ds.get_item(ix)
            sid = str(item["id"])
            ids.append(sid)
            mp[sid] = ix
        return ids, mp

    perc_ids, perc_mp = build_id_map(mv, perception_idxs)
    geom_ids, geom_mp = build_id_map(mv, geometry_idxs)
    hard_ids = [str(m500.get_item(i)["id"]) for i in hard_math_idxs]
    hard_mp = {str(m500.get_item(i)["id"]): i for i in hard_math_idxs}

    chosen_perc = ensure_split_file(
        run_dir / "splits" / f"perception_seed{config.sample_seed}_n{config.sample_size}.jsonl",
        perc_ids, config.sample_size, config.sample_seed
    )
    chosen_geom = ensure_split_file(
        run_dir / "splits" / f"geometry_seed{config.sample_seed}_n{config.sample_size}.jsonl",
        geom_ids, config.sample_size, config.sample_seed
    )
    chosen_hard = ensure_split_file(
        run_dir / "splits" / f"hard_math_seed{config.sample_seed}_n{config.sample_size}.jsonl",
        hard_ids, config.sample_size, config.sample_seed
    )

    print(f"\n  Sampled: {len(chosen_perc)} Perception, {len(chosen_geom)} Geometry, {len(chosen_hard)} Hard Math")

    return {
        "pure_perception": {
            "dataset": mv,
            "indices": [perc_mp[s] for s in chosen_perc if s in perc_mp],
            "settings": [
                ("A_image_prompt", "image_question"),
                ("B_black_image_prompt", "black_image_question"),
            ],
            "type": "perception",
            "description": "Pure visual perception (charts, tables, natural images)",
        },
        "vision_reasoning": {
            "dataset": mv,
            "indices": [geom_mp[s] for s in chosen_geom if s in geom_mp],
            "settings": [
                ("A_image_prompt", "image_question"),
                ("B_black_desc_prompt", "black_image_description_question"),
            ],
            "type": "alignment",
            "description": "Vision + Reasoning (geometry problem solving)",
        },
        "pure_reasoning": {
            "dataset": m500,
            "indices": [hard_mp[s] for s in chosen_hard if s in hard_mp],
            "settings": [
                ("A_prompt_only", "question_only"),
            ],
            "type": "reasoning",
            "description": "Pure reasoning (MATH-500 Level 4-5)",
        },
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(config: EvalConfig):
    """Run the behavior location experiment."""

    # Setup directories
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"behavior_location_{config.experiment_type}_{ts}"
    run_dir = Path(config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    (run_dir / "splits").mkdir(exist_ok=True)
    (run_dir / "cache").mkdir(exist_ok=True)

    # Handle reuse_from: copy cache and splits from previous run
    reuse_dir = None
    if config.reuse_from:
        reuse_dir = Path(config.output_dir) / config.reuse_from
        if reuse_dir.exists():
            print(f"Reusing data from: {reuse_dir}")
            if (reuse_dir / "cache").exists():
                for f in (reuse_dir / "cache").glob("**/*"):
                    if f.is_file():
                        dest = run_dir / "cache" / f.relative_to(reuse_dir / "cache")
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists():
                            shutil.copy2(f, dest)
            if (reuse_dir / "splits").exists():
                for f in (reuse_dir / "splits").glob("*.jsonl"):
                    dest = run_dir / "splits" / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
            print(f"  Copied cache and splits from previous run")
        else:
            print(f"Warning: reuse_from directory not found: {reuse_dir}")

    # Save config
    (run_dir / "config.json").write_text(json.dumps(vars(config), indent=2, default=str))

    # Setup GPT helper
    gpt = None
    if config.use_gpt_descriptions or config.use_gpt_grader:
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai package is required when --use_gpt_descriptions or --use_gpt_grader is set. "
                "Install it with: pip install openai"
            )
        gpt = GPTHelper(model=config.openai_model, cache_dir=str(run_dir / "cache"))
        print(f"GPT helper initialized with model: {config.openai_model}")

    # Load datasets
    print("\n" + "="*60)
    print("Loading Datasets")
    print("="*60)

    mv = MathVistaDataset(split="testmini", cache_path=str(run_dir / "cache" / "mathvista.json"))
    m500 = MATH500Dataset(split="test", cache_path=str(run_dir / "cache" / "math500.json"))

    # Get experiments based on experiment_type
    print(f"\nExperiment type: {config.experiment_type}")
    if config.experiment_type == "v1":
        experiments = get_experiments_v1(mv, m500, config, run_dir)
    else:  # v2 (default)
        experiments = get_experiments_v2(mv, m500, config, run_dir)

    # Get models to test
    models = config.models or [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "OpenMMReasoner/OpenMMReasoner-ColdStart",
        "OpenMMReasoner/OpenMMReasoner-RL",
    ]

    mm = ModelManager(config)
    log_path = run_dir / "logs" / "experiment.log"

    # Main experiment loop
    with open(log_path, "a") as flog:
        def log(msg):
            print(msg)
            flog.write(msg + "\n")
            flog.flush()

        log(f"\n{'='*60}")
        log(f"Behavior Location Experiment (Merged)")
        log(f"{'='*60}")
        log(f"Run: {run_name}")
        log(f"Experiment type: {config.experiment_type}")
        log(f"Models: {len(models)}")
        log(f"Sample size: {config.sample_size}")
        log(f"Seed: {config.sample_seed}")
        log(f"GPT descriptions: {config.use_gpt_descriptions}")
        log(f"GPT grading: {config.use_gpt_grader}")
        if config.reuse_from:
            log(f"Reusing from: {config.reuse_from}")

        for model_name in models:
            log(f"\n{'='*60}")
            log(f"Model: {model_name}")
            log(f"{'='*60}")

            model_dir = run_dir / "results" / sanitize_model_name(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)

            # Check if we can reuse results from previous run
            reuse_model_dir = None
            if reuse_dir and (reuse_dir / "results" / sanitize_model_name(model_name)).exists():
                reuse_model_dir = reuse_dir / "results" / sanitize_model_name(model_name)
                log(f"  Found previous results to reuse")

            # Check if all experiments are already complete
            all_complete = True
            for exp_name in experiments:
                summary_path = model_dir / exp_name / "summary.json"
                if not summary_path.exists():
                    all_complete = False
                    break

            if all_complete:
                log(f"  All experiments already complete, skipping model load")
                continue

            model, model_type = mm.get_model(model_name)

            # Create appropriate evaluator based on model type
            if model_type == "gpt":
                evaluator = GPTEvaluator(
                    model=model_name,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    max_image_pixels=config.max_image_pixels,
                )
            elif model_type == "claude":
                evaluator = ClaudeEvaluator(
                    model=model_name,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    max_image_pixels=config.max_image_pixels,
                )
            else:
                evaluator = HFEvaluator(
                    model=model,
                    processor=mm.processor,
                    model_type=model_type,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    max_image_pixels=config.max_image_pixels,
                    inference_timeout=config.inference_timeout,
                )

            for exp_name, exp_config in experiments.items():
                exp_dir = model_dir / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)

                # Check if summary already exists and is valid
                summary_path = exp_dir / "summary.json"
                if summary_path.exists():
                    try:
                        existing_summary = json.loads(summary_path.read_text())
                        expected_n = len(exp_config["indices"])
                        actual_n = existing_summary.get("N", 0)
                        if actual_n > 0 or expected_n == 0:
                            log(f"\n  Experiment: {exp_name} - ALREADY COMPLETE, skipping")
                            continue
                        else:
                            log(f"\n  Experiment: {exp_name} - Previous summary has N=0 (expected {expected_n}), re-running")
                            summary_path.unlink()
                    except (json.JSONDecodeError, OSError):
                        log(f"\n  Experiment: {exp_name} - Corrupt summary, re-running")
                        summary_path.unlink()

                log(f"\n  Experiment: {exp_name}")
                log(f"    {exp_config['description']}")
                log(f"    Samples: {len(exp_config['indices'])}")

                ds = exp_config["dataset"]
                indices = exp_config["indices"]
                correctness = {}

                for setting_name, input_type in exp_config["settings"]:
                    log(f"    Setting: {setting_name}")

                    out_path = exp_dir / f"{setting_name}.jsonl"
                    done = {}

                    # Load existing results from current run
                    if out_path.exists():
                        for ln in out_path.read_text().splitlines():
                            if ln.strip():
                                try:
                                    rec = json.loads(ln)
                                    done[str(rec["id"])] = rec
                                except:
                                    pass
                        log(f"      Loaded {len(done)} existing results from current run")

                    # Load existing results from reuse_from run
                    if reuse_model_dir:
                        reuse_path = reuse_model_dir / exp_name / f"{setting_name}.jsonl"
                        if reuse_path.exists():
                            reuse_count = 0
                            for ln in reuse_path.read_text().splitlines():
                                if ln.strip():
                                    try:
                                        rec = json.loads(ln)
                                        sid = str(rec["id"])
                                        if sid not in done:
                                            done[sid] = rec
                                            reuse_count += 1
                                    except:
                                        pass
                            if reuse_count > 0:
                                log(f"      Reused {reuse_count} results from previous run")

                    label_map = {}

                    # Fail fast: if this setting requires descriptions but GPT is unavailable
                    if input_type == "black_image_description_question" and gpt is None:
                        log(f"      ERROR: Setting '{setting_name}' requires GPT descriptions but GPT helper is not available.")
                        log(f"      Install openai (pip install openai) and set OPENAI_API_KEY, or use --use_gpt_descriptions.")
                        raise RuntimeError(
                            f"Setting '{setting_name}' requires GPT image descriptions but gpt helper is None. "
                            f"Install openai and ensure OPENAI_API_KEY is set."
                        )

                    with open(out_path, "a") as fout:
                        for ix in tqdm(indices, desc=f"      {setting_name}", leave=False):
                            item = ds.get_item(ix)
                            sid = str(item["id"])

                            # Use existing result
                            if sid in done:
                                rec = done[sid]
                                primary = rec.get("correct_gpt") if rec.get("correct_gpt") is not None else rec.get("correct", False)
                                label_map[sid] = bool(primary)
                                continue

                            # Generate description if needed
                            if input_type == "black_image_description_question":
                                if not item.get("description") and config.use_gpt_descriptions and gpt and item.get("image"):
                                    item["description"] = gpt.describe_image(f"{exp_name}_{sid}", item["image"])

                            # Evaluate
                            result = evaluator.evaluate_single(item, input_type)
                            if result is None:
                                if input_type == "black_image_description_question":
                                    has_desc = bool(item.get("description"))
                                    has_img = item.get("image") is not None
                                    print(f"      [SKIP] id={sid}: evaluate_single returned None "
                                          f"(has_image={has_img}, has_description={has_desc})")
                                continue

                            result["id"] = sid
                            result["dataset"] = exp_name
                            result["model"] = model_name
                            result["setting"] = setting_name

                            # GPT grading
                            if config.use_gpt_grader and gpt:
                                grade = gpt.grade_answer(
                                    f"{exp_name}_{sid}_{sanitize_model_name(model_name)}_{setting_name}",
                                    item["question"],
                                    item.get("choices"),
                                    str(item["answer"]),
                                    str(result["predicted"])
                                )
                                result["correct_gpt"] = bool(grade.get("is_correct", False))
                                result["gpt_grade"] = grade
                            else:
                                result["correct_gpt"] = None

                            primary = result["correct_gpt"] if result["correct_gpt"] is not None else result["correct"]
                            result["correct_primary"] = bool(primary)
                            label_map[sid] = bool(primary)

                            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                            fout.flush()

                    correctness[setting_name] = label_map

                    if label_map:
                        acc = sum(label_map.values()) / len(label_map)
                        log(f"      Accuracy: {acc:.2%} ({sum(label_map.values())}/{len(label_map)})")

                # Compute and save summary
                if exp_config["type"] in ("perception", "alignment"):
                    setting_a = exp_config["settings"][0][0]
                    setting_b = exp_config["settings"][1][0]
                    labels_a = correctness.get(setting_a, {})
                    labels_b = correctness.get(setting_b, {})

                    summary = compute_transition_stats(labels_a, labels_b)
                    summary["setting_A"] = setting_a
                    summary["setting_B"] = setting_b
                    summary["experiment"] = exp_name
                    summary["model"] = model_name
                    summary["description"] = exp_config["description"]

                    # Paper metrics: M_vis and M_v2r
                    N = summary["N"]
                    if exp_config["type"] == "perception":
                        # M_vis = I[f(i,p)=y ∧ f(b,p)≠y] = n_TF / N
                        summary["M_vis"] = round(summary["n_TF"] / N, 4) if N else 0.0
                    elif exp_config["type"] == "alignment":
                        # M_v2r = I[f(i,p)=y ∧ f(b,d,p)=y] = n_TT / N
                        summary["M_v2r"] = round(summary["n_TT"] / N, 4) if N else 0.0

                    log(f"    Summary:")
                    log(f"      Acc A (Image+Prompt): {summary['acc_A']:.2%}")
                    log(f"      Acc B (Control): {summary['acc_B']:.2%}")
                    log(f"      Delta (A-B): {summary['delta_acc']:+.2%}")
                    log(f"      T->F: {summary['n_TF']}, F->T: {summary['n_FT']}")
                    log(f"      Net gain: {summary['net_gain']}")
                    if "M_vis" in summary:
                        log(f"      M_vis: {summary['M_vis']:.4f}")
                    if "M_v2r" in summary:
                        log(f"      M_v2r: {summary['M_v2r']:.4f}")

                else:  # reasoning
                    setting_a = exp_config["settings"][0][0]
                    labels_a = correctness.get(setting_a, {})
                    N = len(labels_a)
                    acc = sum(labels_a.values()) / N if N else 0.0

                    summary = {
                        "N": N,
                        "acc": round(acc, 4),
                        "correct": sum(labels_a.values()),
                        "setting": setting_a,
                        "experiment": exp_name,
                        "model": model_name,
                        "description": exp_config["description"],
                        # M_rea = I[f(p)=y] = acc
                        "M_rea": round(acc, 4),
                    }

                    log(f"    Summary:")
                    log(f"      Accuracy: {acc:.2%} ({summary['correct']}/{N})")
                    log(f"      M_rea: {summary['M_rea']:.4f}")

                (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

            # Cleanup model (skip for API-based models)
            if model_type not in ("gpt", "claude"):
                mm.cleanup(model)

        # ── Final aggregation: M_vis / M_v2r / M_rea per model ──
        log(f"\n{'='*60}")
        log("Paper Metrics Summary (M_vis / M_v2r / M_rea)")
        log(f"{'='*60}")
        metric_table = {}  # model_name -> {M_vis, M_v2r, M_rea}
        for model_name in config.models:
            row = {}
            for exp_name, exp_config in experiments.items():
                exp_dir = run_dir / "results" / model_name / exp_name
                summary_path = exp_dir / "summary.json"
                if summary_path.exists():
                    s = json.loads(summary_path.read_text())
                    for key in ("M_vis", "M_v2r", "M_rea"):
                        if key in s:
                            row[key] = s[key]
            metric_table[model_name] = row

        # Header
        log(f"  {'Model':<55s}  {'M_vis':>7s}  {'M_v2r':>7s}  {'M_rea':>7s}")
        log(f"  {'-'*55}  {'-'*7}  {'-'*7}  {'-'*7}")
        for model_name, row in metric_table.items():
            m_vis = f"{row['M_vis']:.4f}" if "M_vis" in row else "   -  "
            m_v2r = f"{row['M_v2r']:.4f}" if "M_v2r" in row else "   -  "
            m_rea = f"{row['M_rea']:.4f}" if "M_rea" in row else "   -  "
            log(f"  {model_name:<55s}  {m_vis:>7s}  {m_v2r:>7s}  {m_rea:>7s}")
        log("")

        # Save aggregated metrics table
        agg_path = run_dir / "paper_metrics.json"
        agg_path.write_text(json.dumps(metric_table, indent=2, ensure_ascii=False))
        log(f"Aggregated paper metrics saved to: {agg_path}")

        log(f"\n{'='*60}")
        log("Experiment Complete")
        log(f"Results saved to: {run_dir}")
        log(f"{'='*60}")


# ============================================================================
# Re-grading
# ============================================================================

def regrade_run(run_dir: str, openai_model: str = "gpt-4o-mini"):
    """Re-apply GPT grading to all existing results in a run directory.

    This re-grades results that have correct_gpt=null, rewrites the JSONL files,
    deletes stale summaries, and recomputes them.
    No GPU or model inference needed — only OpenAI API calls.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is required for regrading. Install with: pip install openai")

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    gpt = GPTHelper(model=openai_model, cache_dir=str(run_path / "cache"))
    print(f"GPT helper initialized with model: {openai_model}")

    results_path = run_path / "results"
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    total_regraded = 0
    total_flipped = 0

    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir():
            continue
        print(f"\n{'='*60}")
        print(f"Model: {model_dir.name}")
        print(f"{'='*60}")

        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            for jsonl_path in sorted(exp_dir.glob("*.jsonl")):
                records = []
                needs_regrade = 0

                for line in jsonl_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

                if not records:
                    continue

                # Count how many need regrading
                for rec in records:
                    if rec.get("correct_gpt") is None:
                        needs_regrade += 1

                if needs_regrade == 0:
                    print(f"  {exp_dir.name}/{jsonl_path.name}: {len(records)} records, all already graded")
                    continue

                print(f"  {exp_dir.name}/{jsonl_path.name}: {len(records)} records, {needs_regrade} need regrading")

                regraded = 0
                flipped = 0
                for rec in tqdm(records, desc=f"    Grading", leave=False):
                    if rec.get("correct_gpt") is not None:
                        continue

                    cache_key = f"{rec.get('dataset', exp_dir.name)}_{rec['id']}_{sanitize_model_name(rec.get('model', model_dir.name))}_{rec.get('setting', jsonl_path.stem)}"
                    grade = gpt.grade_answer(
                        cache_key,
                        rec.get("question", ""),
                        rec.get("choices"),
                        str(rec.get("gold_answer", "")),
                        str(rec.get("predicted", ""))
                    )
                    rec["correct_gpt"] = bool(grade.get("is_correct", False))
                    rec["gpt_grade"] = grade

                    old_primary = rec.get("correct_primary", rec.get("correct", False))
                    new_primary = rec["correct_gpt"]
                    rec["correct_primary"] = new_primary

                    if bool(old_primary) != bool(new_primary):
                        flipped += 1
                    regraded += 1

                # Rewrite the JSONL file
                with open(jsonl_path, "w") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                total_regraded += regraded
                total_flipped += flipped
                print(f"    Regraded: {regraded}, Flipped: {flipped} (correct<->incorrect)")

            # Delete summary so it gets recomputed
            summary_path = exp_dir / "summary.json"
            if summary_path.exists():
                summary_path.unlink()
                print(f"  Deleted stale summary: {exp_dir.name}/summary.json")

    print(f"\n{'='*60}")
    print(f"Regrading complete")
    print(f"  Total regraded: {total_regraded}")
    print(f"  Total flipped: {total_flipped}")
    print(f"{'='*60}")
    print(f"\nNote: All summary.json files have been deleted.")
    print(f"Re-run the experiment to recompute summaries (existing inference results will be reused).")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation (Merged)")

    parser.add_argument("--sft_model", type=str, default=None,
                       help="Path to SFT/ColdStart model (required unless --regrade)")
    parser.add_argument("--rl_model", type=str, default=None,
                       help="Path to RL/Final model (required unless --regrade)")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--sample_seed", type=int, default=123)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--models", type=str, nargs="+", default=None)

    # GPT options
    parser.add_argument("--use_gpt_descriptions", action="store_true")
    parser.add_argument("--no_gpt_descriptions", action="store_true")
    parser.add_argument("--use_gpt_grader", action="store_true")
    parser.add_argument("--no_gpt_grader", action="store_true")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")

    # Reuse results from previous run
    parser.add_argument("--reuse_from", type=str, default=None,
                       help="Name of previous run to reuse cache/results from")

    # Timeout for single sample inference
    parser.add_argument("--inference_timeout", type=int, default=1200,
                       help="Timeout for single sample inference in seconds (default: 1200 = 20 minutes)")

    # Experiment type: v1 (general) or v2 (finer splits)
    parser.add_argument("--experiment_type", type=str, default="v2", choices=["v1", "v2"],
                       help="Experiment type: v1 (general/math VQA) or v2 (finer perception/geometry splits)")

    # Re-grading mode
    parser.add_argument("--regrade", type=str, default=None, metavar="RUN_DIR",
                       help="Re-grade existing results with GPT (no inference needed). "
                            "Pass the run directory path, e.g. ./runs/eval_v1_mmr1")

    args = parser.parse_args()

    # Re-grading mode: only needs openai, no GPU
    if args.regrade:
        regrade_run(args.regrade, openai_model=args.openai_model)
        return

    # Validate required args for normal mode
    if not args.sft_model or not args.rl_model:
        parser.error("--sft_model and --rl_model are required (unless using --regrade)")

    config = EvalConfig(
        sft_model=args.sft_model,
        rl_model=args.rl_model,
        output_dir=args.output_dir,
        run_name=args.run_name,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        models=args.models,
        use_gpt_descriptions=args.use_gpt_descriptions and not args.no_gpt_descriptions,
        use_gpt_grader=args.use_gpt_grader and not args.no_gpt_grader,
        openai_model=args.openai_model,
        reuse_from=args.reuse_from,
        inference_timeout=args.inference_timeout,
        experiment_type=args.experiment_type,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
