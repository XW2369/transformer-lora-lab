"""ChnSentiCorp 中文情感二分类：baseline -> LoRA 微调 -> 前后对比。"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformerlab.lora import (
    inject_lora,
    lora_param_ratio,
    mark_only_lora_as_trainable,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
MODEL = "bert-base-chinese"


def load_data(n_train, n_test, seed=42):
    train_df = pd.read_parquet(RAW / "chnsenti-train.parquet")
    val_df = pd.read_parquet(RAW / "chnsenti-validation.parquet")   # test split 是空的，用 validation
    train_s, _ = train_test_split(
        train_df, train_size=n_train, stratify=train_df["label"], random_state=seed
    )
    test_s, _ = train_test_split(
        val_df, train_size=n_test, stratify=val_df["label"], random_state=seed
    )
    return train_s.reset_index(drop=True), test_s.reset_index(drop=True)


def encode(tokenizer, texts, labels, max_len):
    enc = tokenizer(
        list(texts), truncation=True, max_length=max_len,
        padding="max_length", return_tensors="pt",
    )
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(list(labels)))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for input_ids, attention_mask, labels in loader:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
    return correct / total


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = n = 0
    for step, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        n += labels.numel()
        if step % 10 == 0:
            print(f"  step {step}/{len(loader)}  loss {loss.item():.4f}", flush=True)
    return total_loss / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    train_df, test_df = load_data(args.n_train, args.n_test, args.seed)
    train_loader = DataLoader(
        encode(tok, train_df["text"], train_df["label"], args.max_len),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        encode(tok, test_df["text"], test_df["label"], args.max_len),
        batch_size=args.batch_size,
    )
    print(f"train={len(train_df)} test={len(test_df)} steps/epoch={len(train_loader)}")

    t0 = time.time()
    acc_before = evaluate(model, test_loader)
    print(f"[baseline] acc_before = {acc_before:.4f}")

    model, cnt = inject_lora(model, r=args.r, alpha=args.alpha)
    mark_only_lora_as_trainable(model)
    trainable, total, ratio = lora_param_ratio(model)
    print(f"[lora] replaced={cnt} trainable={trainable} ratio={ratio*100:.4f}%")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    history = []
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer)
        acc = evaluate(model, test_loader)
        history.append({"epoch": ep, "loss": round(loss, 4), "acc": round(acc, 4)})
        print(f"[epoch {ep}] loss={loss:.4f} acc={acc:.4f}", flush=True)


    acc_after = evaluate(model, test_loader)
    elapsed = time.time() - t0
    print(f"[result] before={acc_before:.4f} after={acc_after:.4f} delta={acc_after-acc_before:+.4f}")
    print(f"[time] {elapsed/60:.1f} min")

    REPORTS.mkdir(parents=True, exist_ok=True)
    result = {
        "model": MODEL,
        "total_params": total,
        "lora_r": args.r,
        "lora_alpha": args.alpha,
        "trainable_params": trainable,
        "trainable_ratio": ratio,
        "replaced_modules": cnt,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "history": history,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "acc_delta": acc_after - acc_before,
        "elapsed_sec": round(elapsed, 1),
    }
    (REPORTS / "lora_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = (
        "# LoRA fine-tuning report\n\n"
        f"- Base model: `{MODEL}` (total params {total:,})\n"
        f"- LoRA r={args.r}, alpha={args.alpha}; replaced {cnt} Linear layers (query/value)\n"
        f"- Trainable params: **{trainable:,}** ({ratio*100:.4f}%)\n"
        f"- Data: train {args.n_train} / test {args.n_test} / max_len {args.max_len} / epochs {args.epochs}\n\n"
        "| metric | value |\n|---|---|\n"
        f"| accuracy before | {acc_before:.4f} |\n"
        f"| accuracy after | {acc_after:.4f} |\n"
        f"| delta | {acc_after-acc_before:+.4f} |\n"
        f"| elapsed | {elapsed/60:.1f} min |\n"
    )
    (REPORTS / "lora_compare.md").write_text(md, encoding="utf-8")
    print("reports written -> reports/lora_result.json, reports/lora_compare.md")


if __name__ == "__main__":
    main()
