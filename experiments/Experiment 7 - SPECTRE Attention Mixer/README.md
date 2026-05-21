# Experiment 7 - SPECTRE Attention Mixer

## Goal

Test SPECTRE as an attention replacement in the local HRM-Text probe.

This is different from the earlier Fourier weight runs. The earlier work changed how projection weights are stored. This experiment changes the token mixer: softmax attention becomes a small FFT-based mixer with learned complex frequency gates.

## What Changed

- Added `SpectreAttention` as an optional `token_mixer: spectre`.
- Kept regular softmax attention as the default path.
- Made the first version leak-safe for PrefixLM:
  - prefix tokens only mix with prefix tokens
  - response tokens only mix with prefix plus previous/current response tokens
- Cached generation is not implemented yet.

## How To Run

```powershell
rtk python "experiments\Experiment 7 - SPECTRE Attention Mixer\spectre_attention_probe.py" --steps 40 --seeds 1 --device cpu
```

For a slightly stronger local probe:

```powershell
rtk python "experiments\Experiment 7 - SPECTRE Attention Mixer\spectre_attention_probe.py" --steps 80 --seeds 1,2,3 --variants dense,spectre,spectre-fourier-attention,spectre-fourier-all --device cpu --fourier-mode 32
```

## Current Notes

This first implementation is intentionally simple and slow. It still recomputes the FFT per allowed context window so we can prove no future-token leak before deeper optimization.

The first speed pass precomputes SPECTRE `q_proj` and `v_proj` once per packed sequence. That removes repeated projection work while keeping the same allowed context windows.

## First Local Result

Command:

```powershell
rtk python "experiments\Experiment 7 - SPECTRE Attention Mixer\spectre_attention_probe.py" --steps 80 --seeds 1,2,3 --variants dense,spectre,spectre-fourier-attention,spectre-fourier-all --device cpu --fourier-mode 32
```

Result:

| Variant | Mean Eval Loss | Params | Read |
| --- | ---: | ---: | ---: |
| dense | 3.6445 | 172,544 | baseline softmax attention |
| spectre | 3.6303 | 173,064 | slightly better, same parameter scale |
| spectre-fourier-attention | 3.7015 | 154,632 | smaller attention projections, weaker here |
| spectre-fourier-all | 3.5983 | 60,424 | best mean eval here, about 35% of dense params |

Read: SPECTRE trains through the HRM-Text path. The best tiny result here is `spectre-fourier-all`, but this is still a small local probe. Speed is still the weak part: the safe SPECTRE path is about 8-10x slower than dense attention on CPU.
