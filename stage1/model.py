"""
model.py - Stage 1: GPU-enabled global singleton model with batch inference.
Loads trained model if available, else falls back to base model + keyword boosting.
"""

import os
import re
from typing import List, Tuple

TRAINED_MODEL_DIR = "stage1_trained_model"
BASE_MODEL_PRIMARY = "nlpaueb/legal-bert-base-uncased"
BASE_MODEL_FALLBACK = "distilbert-base-uncased"
ID2LABEL = {0: "FACT", 1: "ARGUMENT", 2: "REASONING"}
INFERENCE_BATCH_SIZE = 32
ALLOW_REMOTE_MODEL_DOWNLOAD = os.getenv("ALLOW_REMOTE_MODEL_DOWNLOAD", "0").strip() == "1"

# ── Global singleton ──────────────────────────────────────────────────────────
_tokenizer      = None
_model          = None
_device         = None
_model_available = False
_is_trained     = False

# Keyword boost patterns (used when base model is loaded without fine-tuning)
_REASONING_RE = re.compile(
    r"\b(therefore|thus|hence|accordingly|we\s+find|we\s+hold|court\s+observed"
    r"|in\s+our\s+view|it\s+is\s+clear|we\s+therefore|consequently"
    r"|the\s+court\s+held|this\s+court\s+held|no\s+merit|we\s+are\s+of\s+the\s+view)\b",
    re.IGNORECASE,
)
_ARGUMENT_RE = re.compile(
    r"\b(argued|contended|submitted|pleaded|claimed|according\s+to"
    r"|on\s+behalf\s+of|the\s+defence|the\s+prosecution|counsel\s+for"
    r"|it\s+was\s+argued|it\s+was\s+submitted|the\s+contention|alleged|urged)\b",
    re.IGNORECASE,
)


def load_model() -> bool:
    """
    Load model into global cache (runs only once).
    Priority: trained model → legal-bert → distilbert → keyword-only.
    Moves model to GPU if available.
    """
    global _tokenizer, _model, _device, _model_available, _is_trained

    if _model is not None:
        return _model_available

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Stage 1] Using device: {_device}")

        # Try trained model first
        if os.path.isdir(TRAINED_MODEL_DIR):
            try:
                print(f"[Stage 1] Loading trained model from: {TRAINED_MODEL_DIR}")
                _tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR, local_files_only=True)
                _model     = AutoModelForSequenceClassification.from_pretrained(
                    TRAINED_MODEL_DIR,
                    local_files_only=True,
                )
                _model.to(_device)
                _model.eval()
                _model_available = True
                _is_trained      = True
                print(f"[Stage 1] Trained model loaded on {_device}")
                return True
            except Exception as e:
                print(f"[Stage 1] Could not load trained model: {e}")

        # Fallback to base model
        for model_name in (BASE_MODEL_PRIMARY, BASE_MODEL_FALLBACK):
            try:
                import transformers
                # Suppress "uninitialized weights" warning for base models
                prev_verbosity = transformers.logging.get_verbosity()
                transformers.logging.set_verbosity_error()

                print(f"[Stage 1] Loading base model: {model_name}")
                _tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=not ALLOW_REMOTE_MODEL_DOWNLOAD,
                )
                _model     = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    local_files_only=not ALLOW_REMOTE_MODEL_DOWNLOAD,
                )
                transformers.logging.set_verbosity(prev_verbosity)
                _model.to(_device)
                _model.eval()
                _model_available = True
                _is_trained      = False
                print(f"[Stage 1] Base model loaded on {_device} (keyword boosting active)")
                return True
            except Exception as e:
                print(f"[Stage 1] Could not load {model_name}: {e}")

        _model_available = False
        return False

    except ImportError:
        print("[Stage 1] torch/transformers not installed — rules-only mode.")
        _model_available = False
        return False


def predict_batch(sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Batch inference: tokenize all sentences at once, single forward pass.
    Processes in sub-batches of INFERENCE_BATCH_SIZE to avoid OOM.

    Returns list of (label, confidence) tuples.
    """
    global _tokenizer, _model, _device, _model_available, _is_trained

    if not _model_available or _model is None or not sentences:
        return [_keyword_fallback(s) for s in sentences]

    try:
        import torch
        import torch.nn.functional as F

        all_results: List[Tuple[str, float]] = []

        for start in range(0, len(sentences), INFERENCE_BATCH_SIZE):
            batch_sents = sentences[start: start + INFERENCE_BATCH_SIZE]

            inputs = _tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {k: v.to(_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = _model(**inputs)

            probs       = F.softmax(outputs.logits, dim=-1)
            pred_ids    = probs.argmax(dim=-1).tolist()
            confidences = probs.max(dim=-1).values.tolist()

            for sent, pred_id, conf in zip(batch_sents, pred_ids, confidences):
                if _is_trained:
                    label = ID2LABEL.get(pred_id, "FACT")
                else:
                    # Base model not fine-tuned — apply keyword boost
                    label = _apply_keyword_boost(sent, float(conf))
                all_results.append((label, float(conf)))

        return all_results

    except Exception as e:
        print(f"[Stage 1] Batch inference error: {e} — keyword fallback.")
        return [_keyword_fallback(s) for s in sentences]


def _apply_keyword_boost(sentence: str, base_conf: float) -> str:
    r = bool(_REASONING_RE.search(sentence))
    a = bool(_ARGUMENT_RE.search(sentence))
    if r and not a:
        return "REASONING"
    if a and not r:
        return "ARGUMENT"
    if r and a:
        return "REASONING"
    return "FACT"


def _keyword_fallback(sentence: str) -> Tuple[str, float]:
    if _REASONING_RE.search(sentence):
        return ("REASONING", 0.75)
    if _ARGUMENT_RE.search(sentence):
        return ("ARGUMENT", 0.70)
    return ("FACT", 0.60)


def is_model_available() -> bool:
    return _model_available


def is_trained_model() -> bool:
    return _is_trained
