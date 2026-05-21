from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dataset_new import V1Dataset, V1DatasetConfig
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.lm_head import LMHead


def build_tiny_dataset(path: Path, *, num_examples: int = 64, inst_len: int = 12, resp_len: int = 12) -> Path:
    if path.exists():
        shutil.rmtree(path)
    (path / "epoch_0").mkdir(parents=True)

    tokens = []
    inst_start, inst_lengths, resp_start, resp_lengths = [], [], [], []
    cursor = 0
    for example_idx in range(num_examples):
        inst = [1 + ((example_idx * 11 + i) % 250) for i in range(inst_len)]
        resp = [1 + ((example_idx * 17 + i * 3 + 7) % 250) for i in range(resp_len)]

        inst_start.append(cursor)
        inst_lengths.append(inst_len)
        tokens.extend(inst)
        cursor += inst_len

        resp_start.append(cursor)
        resp_lengths.append(resp_len)
        tokens.extend(resp)
        cursor += resp_len

    np.save(path / "tokens.npy", np.array(tokens, dtype=np.int32))
    np.save(path / "epoch_0" / "inst_start.npy", np.array(inst_start, dtype=np.int64))
    np.save(path / "epoch_0" / "inst_len.npy", np.array(inst_lengths, dtype=np.int64))
    np.save(path / "epoch_0" / "resp_start.npy", np.array(resp_start, dtype=np.int64))
    np.save(path / "epoch_0" / "resp_len.npy", np.array(resp_lengths, dtype=np.int64))

    metadata = {
        "tokenizer_info": {"vocab_size": 260},
        "vocab_size": None,
        "max_seq_len": inst_len + resp_len,
        "total_length": num_examples * (inst_len + resp_len - 1),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_one_v1_batch(dataset_path: Path, *, batch_max_length: int):
    dataset = V1Dataset(V1DatasetConfig(
        seed=0,
        dataset_path=str(dataset_path),
        batch_max_length=batch_max_length,
        drop_last_batch=False,
        target_only=True,
        rank=0,
        num_replicas=1,
    ))
    batch, scalars = next(iter(dataset))
    return batch, scalars, dataset.metadata


def load_v1_batches(dataset_path: Path, *, batch_max_length: int):
    dataset = V1Dataset(V1DatasetConfig(
        seed=0,
        dataset_path=str(dataset_path),
        batch_max_length=batch_max_length,
        drop_last_batch=False,
        target_only=True,
        rank=0,
        num_replicas=1,
    ))
    batches = list(iter(dataset))
    return batches, dataset.metadata


def make_config(*, fourier: bool, vocab_size: int, max_seq_len: int, hidden_size: int, mode: int) -> dict:
    config = {
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "n_layers": 2,
        "hidden_size": hidden_size,
        "num_heads": 4,
        "expansion": 2,
        "attn_type": "prefixlm",
        "init_type": "lecun_normal",
        "norm_type": "pre",
        "norm_eps": 1e-6,
        "pos_emb_type": "none",
        "half_layers": True,
        "H_cycles": 1,
        "L_cycles": 1,
        "bp_warmup_ratio": 0.0,
        "bp_min_steps": 2,
        "bp_max_steps": 2,
        "H_override": {},
    }
    if fourier:
        config["fourier_linear"] = {
            "enabled": True,
            "target": "mlp",
            "in_modes": mode,
            "out_modes": mode,
        }
    return config


def move_batch(batch: dict[str, torch.Tensor], scalars: dict[str, int], device: torch.device) -> dict:
    moved = {name: tensor.to(device) for name, tensor in batch.items()}
    moved.update({name: torch.tensor(value, device=device) for name, value in scalars.items()})
    return moved


def train_variant(name: str,
                  *,
                  fourier: bool,
                  batches,
                  metadata,
                  steps: int,
                  device: torch.device,
                  hidden_size: int,
                  mode: int) -> dict[str, float]:
    torch.manual_seed(19)
    model = LMHead(
        HierarchicalReasoningModel(make_config(fourier=fourier, vocab_size=metadata.vocab_size, max_seq_len=metadata.max_seq_len, hidden_size=hidden_size, mode=mode)),
        {"vocab_size": metadata.vocab_size},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    first_loss = None
    final_loss = None
    for step in range(steps):
        batch, scalars = batches[step % len(batches)]
        batch = move_batch(batch, scalars, device)
        optimizer.zero_grad(set_to_none=True)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = final_loss

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    return {
        "first_loss": first_loss or 0.0,
        "final_loss": final_loss or 0.0,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": time.perf_counter() - start,
    }


def format_result_row(name: str, result: dict[str, float]) -> str:
    return (
        f"{name}: "
        f"loss {result['first_loss']:.4f} -> {result['final_loss']:.4f}, "
        f"params={int(result['num_params']):,}, "
        f"peak_vram_mb={result['peak_vram_mb']:.1f}, "
        f"elapsed_s={result['elapsed_s']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny local HRM-Text V1Dataset-format probe.")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--mode", type=int, default=64)
    parser.add_argument("--batch-max-length", type=int, default=256)
    parser.add_argument("--num-examples", type=int, default=96)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_path = build_tiny_dataset(Path(__file__).with_name("tiny_dataset"), num_examples=args.num_examples)
    batches, metadata = load_v1_batches(dataset_path, batch_max_length=args.batch_max_length)

    print(f"device={device}")
    print(f"dataset={dataset_path}")
    print(f"batches={len(batches)}, vocab_size={metadata.vocab_size}, max_seq_len={metadata.max_seq_len}, steps={args.steps}")
    dense = train_variant("dense", fourier=False, batches=batches, metadata=metadata, steps=args.steps, device=device, hidden_size=args.hidden_size, mode=args.mode)
    print(format_result_row("dense", dense))
    fourier = train_variant(f"fourier-{args.mode}", fourier=True, batches=batches, metadata=metadata, steps=args.steps, device=device, hidden_size=args.hidden_size, mode=args.mode)
    print(format_result_row(f"fourier-{args.mode}", fourier))


if __name__ == "__main__":
    main()
