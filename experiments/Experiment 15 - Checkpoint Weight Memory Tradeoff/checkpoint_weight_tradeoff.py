from pathlib import Path
import sys
import time
import gc

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lm_head import LMHead

LOCKED_CHECKPOINT_WEIGHT = True
LOCKED_CONTROL_CHECKPOINT_WEIGHT = False


class DummyModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head_hint = {
            "in": {"dim": hidden_size, "init_std": 0.02},
            "out": {"dim": hidden_size, "init_std": 0.02},
        }
        self.create_cache = lambda **_kwargs: None
        self.compute_train_extra_args = lambda _state: {}

    def forward(self, carry, x, **_seq_info):
        # Simply return hidden states unchanged
        return carry, x

def run_benchmark(checkpoint_weight: bool, device: str = "cuda", num_steps: int = 10):
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, running on CPU")
        device = "cpu"
        
    device = torch.device(device)
    
    vocab_size = 50000
    hidden_size = 1536
    vocab_modes = 512
    hidden_modes = 64
    
    batch_size = 4
    seq_len = 1024
    
    torch.manual_seed(42)
    
    # Instantiate the wrapper LMHead model which handles embedding, model forward, and logits/loss
    model = LMHead(
        DummyModel(hidden_size=hidden_size),
        {
            "vocab_size": vocab_size,
            "vocab_head": {
                "type": "tied_fourier",
                "vocab_modes": vocab_modes,
                "hidden_modes": hidden_modes,
                "checkpoint_weight": checkpoint_weight,
            },
        },
    ).to(device)
    model.train()
    
    # Generate batch
    batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size * seq_len,), device=device),
        "labels": torch.randint(0, vocab_size, (batch_size * seq_len,), device=device),
        "cu_seqlens": torch.arange(0, (batch_size + 1) * seq_len, seq_len, device=device),
    }
    
    # Warmup
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        carry, loss, metrics = model(carry=None, batch=batch)
        loss.backward()
        
    # Benchmark
    fwd_times = []
    bwd_times = []
    
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        
    for step in range(num_steps):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        
        t0 = time.perf_counter()
        carry, loss, metrics = model(carry=None, batch=batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)
        
        t2 = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        bwd_times.append(t3 - t2)
        
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0
        
    avg_fwd = sum(fwd_times) / len(fwd_times) * 1000  # ms
    avg_bwd = sum(bwd_times) / len(bwd_times) * 1000  # ms
    
    coef_grad = model.vocab_head.coefficients.grad.clone() if model.vocab_head.coefficients.grad is not None else None
    
    return peak_vram_mb, avg_fwd, avg_bwd, coef_grad

def main():
    print("Running Checkpoint Weight Memory Tradeoff Benchmark...")
    print("Vocab size: 50,000, Hidden size: 1,536")
    print("-" * 75)
    
    # 1. Benchmark Checkpoint OFF
    vram_off, fwd_off, bwd_off, coef_g_off = run_benchmark(checkpoint_weight=False)
    print(f"Checkpoint OFF: Peak VRAM = {vram_off:.2f} MB | Fwd = {fwd_off:.2f} ms | Bwd = {bwd_off:.2f} ms")
    
    # 2. Benchmark Checkpoint ON
    vram_on, fwd_on, bwd_on, coef_g_on = run_benchmark(checkpoint_weight=True)
    print(f"Checkpoint ON:  Peak VRAM = {vram_on:.2f} MB | Fwd = {fwd_on:.2f} ms | Bwd = {bwd_on:.2f} ms")
    
    print("-" * 75)
    # Check gradient similarity
    if coef_g_off is not None and coef_g_on is not None:
        coef_diff = torch.max(torch.abs(coef_g_off - coef_g_on)).item()
        print(f"Max Gradient Difference (coefficients): {coef_diff:.2e}")
        
    print("-" * 75)
    print("VRAM Savings:")
    reduction = vram_off - vram_on
    pct = (reduction / vram_off) * 100
    print(f"Peak VRAM reduction: {reduction:.2f} MB ({pct:.1f}%)")

if __name__ == "__main__":
    main()
