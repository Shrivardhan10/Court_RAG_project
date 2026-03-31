"""
dataset.py - Stage 1: PyTorch Dataset for legal sentence classification.
"""

import json
from typing import List, Tuple

LABEL2ID = {"FACT": 0, "ARGUMENT": 1, "REASONING": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_training_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Load training data from a JSON file.

    Expected format:
        [ {"text": "...", "label": "FACT"}, ... ]

    Returns:
        texts  - list of sentence strings
        labels - list of integer label ids (FACT=0, ARGUMENT=1, REASONING=2)
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    texts: List[str] = []
    labels: List[int] = []

    for rec in records:
        text  = rec.get("text", "").strip()
        label = rec.get("label", "FACT").upper().strip()
        if text and label in LABEL2ID:
            texts.append(text)
            labels.append(LABEL2ID[label])

    return texts, labels


try:
    import torch
    from torch.utils.data import Dataset

    class LegalDataset(Dataset):
        """
        PyTorch Dataset wrapping tokenized legal sentences.

        Args:
            encodings: Output of tokenizer(texts, ...) — a BatchEncoding.
            labels:    List of integer label ids.
        """

        def __init__(self, encodings, labels: List[int]):
            self.encodings = encodings
            self.labels    = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

except ImportError:
    # torch not available — dataset class unavailable but load_training_data still works
    class LegalDataset:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to use LegalDataset")
