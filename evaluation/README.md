## HRM evaluation

Evaluation takes 1 GPU (assmuing 80 GiB VRAM). Lower batch_size when OOM issues occur.

```bash
python -m evaluation.main ckpt_path="<CHECKPOINT_PATH>"
```

### Laptop probe checkpoints (Spectral-HRM experiments)

Experiment probes (Exp 25 / 32) save a lightweight layout under `runs/.../ckpts/<variant>_h<size>_s<seed>/`. Use the probe harness (CUDA, generation benchmarks only):

```bash
python scripts/probe_benchmark.py --ckpt-dir runs/exp32/ckpts/fourier-pom-fla-gdn-projected-dense-tied_h512_s1 --benchmarks GSM8k --limit 50
```

This is **not** comparable to the table below (different scale and training). For official HRM-Text numbers, use `ckpt_path="sapientinc/HRM-Text-1B"` (or your FSDP pretrain dir) with `evaluation.main`.

## Baseline evaluation

```bash
# Llama3.2 3B
python -m evaluation.main ckpt_path="unsloth/Llama-3.2-3B" config="evaluation/config/vllm_benchmarking.yaml"
lm-eval run --model vllm --model_args pretrained=unsloth/Llama-3.2-3B,max_model_len=3072 --tasks minerva_math --gen_kwargs temperature=0.0 --batch_size auto

# Olmo-3 7B
python -m evaluation.main ckpt_path="allenai/Olmo-3-1025-7B" config="evaluation/config/vllm_benchmarking.yaml"

# Qwen-3.5 2B
python -m evaluation.main ckpt_path="Qwen/Qwen3.5-2B" config="evaluation/config/vllm_benchmarking.yaml"
lm-eval run --model vllm --model_args pretrained=Qwen/Qwen3.5-2B,max_model_len=3072 --tasks minerva_math --gen_kwargs temperature=0.0 --batch_size auto

# Ouro 1.4B
python -m evaluation.main ckpt_path="ByteDance/Ouro-1.4B" config="evaluation/config/vllm_benchmarking.yaml" trust_remote_code=True gpu_memory_utilization=0.8
lm-eval run --model vllm --model_args pretrained=ByteDance/Ouro-1.4B,max_model_len=3072,gpu_memory_utilization=0.8,trust_remote_code=True,add_bos_token=True --tasks minerva_math --gen_kwargs temperature=0.0 --batch_size auto
```
