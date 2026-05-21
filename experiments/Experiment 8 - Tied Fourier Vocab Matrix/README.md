# Experiment 8 - Tied Fourier Vocab Matrix

## Goal

Compress the text-specific vocab parameters by using one shared Fourier-generated matrix for both input embedding and LM head output.

Instead of:

```text
embedding matrix + separate lm_head matrix
```

this experiment uses:

```text
one Fourier coefficient grid
  -> temporary dense vocab matrix W
  -> embedding uses W[token_id]
  -> LM head uses W.T
```

## What Changed

- Added `TiedFourierVocab` in `models/lm_head.py`.
- Added config switch:

```yaml
vocab_head:
  type: tied_fourier
  vocab_modes: 160
  hidden_modes: 64
```

- Dense vocab behavior remains the default.

## How To Run

```powershell
rtk python "experiments\Experiment 8 - Tied Fourier Vocab Matrix\tied_fourier_vocab_probe.py" --steps 80 --seeds 1,2,3 --device cpu
```

## Current Notes

This reconstructs the dense vocab matrix once per forward pass and reuses it for both embedding lookup and logits. During training autograd keeps it for backward, so it is temporary compute memory, not a persistent trainable dense parameter.

## First Local Result

Command:

```powershell
rtk python "experiments\Experiment 8 - Tied Fourier Vocab Matrix\tied_fourier_vocab_probe.py" --steps 80 --seeds 1,2,3 --device cpu --vocab-modes 160 --hidden-modes 64 --fourier-mode 32
```

Result:

| Variant | Mean Eval Loss | Params | Read |
| --- | ---: | ---: | --- |
| dense | 3.6445 | 172,544 | baseline untied vocab |
| tied-fourier-vocab | 3.6291 | 149,504 | slightly better here, modest param cut |
| fourier-all | 3.6014 | 41,472 | best tiny eval here |
| fourier-all-tied-fourier-vocab | 3.7468 | 18,432 | huge param cut, some quality loss |

Read: `tied-fourier-vocab` works as a safe first vocab compression. Combining it with `fourier-all` is very small, but this local mode is probably too compressed for quality.
