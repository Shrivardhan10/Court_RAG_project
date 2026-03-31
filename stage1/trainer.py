"""
trainer.py - Stage 1: Fine-tune BERT on FACT / ARGUMENT / REASONING labels.

Usage:
    python -m stage1.trainer --data path/to/train.json --epochs 3 --batch_size 16

Training data format (JSON):
    [ {"text": "...", "label": "FACT"}, ... ]

Output:
    ./stage1_trained_model/   (tokenizer + model weights)
"""

import os
import argparse

SAVE_DIR = "stage1_trained_model"
BASE_MODEL = "nlpaueb/legal-bert-base-uncased"
FALLBACK_MODEL = "distilbert-base-uncased"
NUM_LABELS = 3


def train_model(
    data_path: str,
    epochs: int = 3,
    batch_size: int = 16,
    save_dir: str = SAVE_DIR,
    base_model: str = BASE_MODEL,
) -> None:
    """
    Fine-tune a BERT model for 3-class legal sentence classification.

    Args:
        data_path:  Path to training JSON file.
        epochs:     Number of training epochs.
        batch_size: Training batch size.
        save_dir:   Directory to save the trained model.
        base_model: HuggingFace model name to fine-tune.
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    from stage1.dataset import load_training_data, LegalDataset, ID2LABEL, LABEL2ID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[Trainer] Loading training data from: {data_path}")
    texts, labels = load_training_data(data_path)
    print(f"[Trainer] Samples: {len(texts)} | Labels: {dict(zip(*zip(*[(ID2LABEL[l], labels.count(l)) for l in set(labels)])))}")

    # ── Load tokenizer + model ────────────────────────────────────────────────
    for model_name in (base_model, FALLBACK_MODEL):
        try:
            print(f"[Trainer] Loading base model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=NUM_LABELS,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            )
            print(f"[Trainer] Base model loaded: {model_name}")
            break
        except Exception as e:
            print(f"[Trainer] Could not load {model_name}: {e}")
    else:
        raise RuntimeError("[Trainer] No base model could be loaded.")

    model.to(device)

    # ── Tokenize ──────────────────────────────────────────────────────────────
    print("[Trainer] Tokenizing ...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    dataset    = LegalDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn   = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct    = 0
        total      = 0

        for step, batch in enumerate(dataloader, start=1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids      = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, label_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = outputs.logits.argmax(dim=-1)
            correct    += (preds == label_ids).sum().item()
            total      += len(label_ids)

            if step % 20 == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(dataloader)} | "
                      f"Loss: {total_loss/step:.4f} | Acc: {correct/total:.3f}")

        print(f"[Trainer] Epoch {epoch} done | "
              f"Avg Loss: {total_loss/len(dataloader):.4f} | "
              f"Accuracy: {correct/total:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[Trainer] Model saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 classifier")
    parser.add_argument("--data",       required=True, help="Path to training JSON")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir",   default=SAVE_DIR)
    parser.add_argument("--base_model", default=BASE_MODEL)
    args = parser.parse_args()

    train_model(
        data_path  = args.data,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        save_dir   = args.save_dir,
        base_model = args.base_model,
    )
