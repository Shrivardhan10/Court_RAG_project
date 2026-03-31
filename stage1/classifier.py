"""
classifier.py - Stage 1: Hybrid classifier (rules + ML model, always both).
Model inference always runs. Rules only override on strong keyword matches.
"""

from typing import List
from stage1.rules import score_sentence

_STRONG_THRESHOLD = 2


def classify_batch(sentences: List[str]) -> List[str]:
    """
    Classify sentences using hybrid rules + model.

    Always runs model inference on ALL sentences.
    Rules override only when score >= STRONG_THRESHOLD.

    Returns list of labels: FACT / ARGUMENT / REASONING
    """
    if not sentences:
        return []

    # Step 1: score all sentences with rules
    rule_scores = [score_sentence(s) for s in sentences]

    # Step 2: always run model on all sentences (no skipping)
    from stage1.model import predict_batch
    model_predictions = predict_batch(sentences)

    # Step 3: combine — strong rule wins, else trust model
    final: List[str] = []
    for i, scores in enumerate(rule_scores):
        r = scores["REASONING"]
        a = scores["ARGUMENT"]
        model_label, model_conf = model_predictions[i]

        if r >= _STRONG_THRESHOLD and r >= a:
            final.append("REASONING")
        elif a >= _STRONG_THRESHOLD and a > r:
            final.append("ARGUMENT")
        else:
            final.append(model_label)

    return final
