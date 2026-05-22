"""Verify V1Dataset layout and print pretrain step estimates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_new import V1Dataset, V1DatasetConfig, V1DatasetIndices  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192, help="global_batch_size for step estimate")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--peek-batches", type=int, default=2, help="Load this many batches (0 = skip)")
    args = parser.parse_args()

    path = args.dataset_path.resolve()
    required = [
        path / "metadata.json",
        path / "tokens.npy",
        path / "epoch_0" / "inst_start.npy",
        path / "epoch_0" / "inst_len.npy",
        path / "epoch_0" / "resp_start.npy",
        path / "epoch_0" / "resp_len.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("MISSING:", *missing, sep="\n  ")
        sys.exit(1)

    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    total_length = int(meta["total_length"])
    tokens = np.load(path / "tokens.npy", mmap_mode="r")
    indices = V1DatasetIndices(
        **{
            f.name: np.load(path / "epoch_0" / f"{f.name}.npy", mmap_mode="r")
            for f in fields(V1DatasetIndices)
        }
    )
    n_examples = len(indices.inst_start)
    steps = int(args.epochs * total_length // args.batch_size)

    print(f"dataset: {path}")
    print(f"  tokens on disk: {tokens.shape[0]:,} int32 ({tokens.nbytes / 2**20:.1f} MiB)")
    print(f"  examples: {n_examples:,}")
    print(f"  total_length (train): {total_length:,}")
    print(f"  max_seq_len (metadata): {meta.get('max_seq_len')}")
    print(f"  vocab (tokenizer_info): {meta.get('tokenizer_info', {}).get('vocab_size')}")
    print(f"  steps @ global_batch_size={args.batch_size}, epochs={args.epochs}: {steps:,}")

    if args.peek_batches > 0:
        ds = V1Dataset(
            V1DatasetConfig(
                seed=0,
                dataset_path=str(path),
                batch_max_length=args.batch_size,
                drop_last_batch=True,
                target_only=True,
                rank=0,
                num_replicas=1,
            )
        )
        it = iter(ds)
        for i in range(args.peek_batches):
            batch, scalars = next(it)
            print(f"  batch {i}: inputs {tuple(batch['inputs'].shape)} loss-scale keys ok")
        print("  V1Dataset iterator: OK")

    print("verify: OK")


if __name__ == "__main__":
    main()
