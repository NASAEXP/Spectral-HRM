# Experiment 13 - Fourier Vocab Ordering

## Goal

Test whether the Fourier vocab grid gets better when token IDs are arranged more like a smooth signal.

The local probe uses byte tokens, so this experiment uses byte-aware orderings first. Later, the same custom permutation hook can use a real tokenizer order from Sapient's BPE vocab.

## What Changed

- Added `TiedFourierVocab.set_token_permutation(...)`.
- Added a local ordering runner.
- Reused the Experiment 8 training probe instead of duplicating the training loop.

## How To Run

```powershell
rtk python "experiments\Experiment 13 - Fourier Vocab Ordering\fourier_vocab_ordering.py" --steps 80 --seeds 1,2,3 --device cuda
```

Fast smoke:

```powershell
rtk python "experiments\Experiment 13 - Fourier Vocab Ordering\fourier_vocab_ordering.py" --steps 20 --seeds 1 --device cuda
```

## Orderings

| Ordering | Meaning |
| --- | --- |
| `identity` | current token ID order |
| `frequency` | most common local tokens first |
| `byte_category` | reserved, whitespace, digits, uppercase, lowercase, punctuation, other ASCII, extended |
| `byte_category_frequency` | byte category first, then most common tokens inside each category |
| `random` | deterministic random control |

## Current Read

### Compressed Body Branch

Command:

```powershell
rtk python "experiments\Experiment 13 - Fourier Vocab Ordering\fourier_vocab_ordering.py" --steps 40 --seeds 1,2,3 --device cuda
```

Base variant: `fourier-all-tied-fourier-vocab-bias`

| Ordering | Mean Eval Loss | Params | Peak VRAM | Read |
| --- | ---: | ---: | ---: | --- |
| `identity` | 4.0093 | 49,668 | 19.2 MB | best in this pass |
| `frequency` | 4.3020 | 49,668 | 19.2 MB | worse |
| `byte_category` | 4.1078 | 49,668 | 19.2 MB | close, but not better |
| `byte_category_frequency` | 4.2109 | 49,668 | 19.2 MB | worse |
| `random` | 4.0462 | 49,668 | 19.2 MB | close to identity, but weaker |

### Vocab-Only Branch

Command:

```powershell
rtk python "experiments\Experiment 13 - Fourier Vocab Ordering\fourier_vocab_ordering.py" --steps 40 --seeds 1,2,3 --device cuda --base-variant tied-fourier-vocab-bias
```

Base variant: `tied-fourier-vocab-bias`

| Ordering | Mean Eval Loss | Params | Peak VRAM | Read |
| --- | ---: | ---: | ---: | --- |
| `identity` | 3.9079 | 377,348 | 23.8 MB | best in this pass |
| `frequency` | 4.0299 | 377,348 | 23.8 MB | worse |
| `byte_category` | 3.9875 | 377,348 | 23.8 MB | weaker |
| `byte_category_frequency` | 4.0152 | 377,348 | 23.8 MB | weaker |
| `random` | 3.9394 | 377,348 | 23.8 MB | close, but weaker |

Read: for the current local byte-token probe, identity order is still the best default. Frequency and category shortcuts did not unlock the Fourier vocab. The hook is still useful, but the next serious ordering should be tokenizer-aware: BPE rank, token text category, or learned/co-occurrence order from the Sapient data.
