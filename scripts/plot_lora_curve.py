"""Plot LoRA fine-tuning results from reports/lora_result.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

data = json.loads((REPORTS / "lora_result.json").read_text(encoding="utf-8"))
history = data["history"]
epochs = [h["epoch"] for h in history]
accs = [h["acc"] for h in history]
losses = [h["loss"] for h in history]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.bar(["before", "after"], [data["acc_before"], data["acc_after"]], color=["#9aa5b1", "#2f7ed8"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("accuracy")
ax.set_title(f"LoRA fine-tuning: +{data['acc_delta'] * 100:.2f} pp")
for i, v in enumerate([data["acc_before"], data["acc_after"]]):
    ax.text(i, v + 0.02, f"{v:.4f}", ha="center")

ax = axes[1]
ax.plot(epochs, accs, marker="o", label="test acc")
ax.plot(epochs, losses, marker="s", linestyle="--", label="train loss")
ax.set_xlabel("epoch")
ax.set_xticks(epochs)
ax.set_title("per-epoch curve")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle(
    f"bert-base-chinese + LoRA(r={data['lora_r']}, alpha={data['lora_alpha']}, "
    f"trainable={data['trainable_ratio'] * 100:.4f}%) | "
    f"n_train={data['n_train']} n_test={data['n_test']}"
)
fig.tight_layout()
out = REPORTS / "figures" / "acc_compare.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("saved ->", out)
