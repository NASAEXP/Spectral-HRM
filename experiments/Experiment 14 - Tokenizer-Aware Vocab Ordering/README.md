# Experiment 14 - Tokenizer-Aware Vocab Ordering

## Goal

Retest Fourier vocab ordering using Sapient's real BPE tokenizer instead of the byte-token shortcut from Experiment 13.

This matters because `identity` in the real tokenizer is already close to BPE-rank order. If any ordering beats it, that is more meaningful than the byte-token result.

## What Changed

- Added a full-vocab runner using `data_io/trained_tokenizers/bpe/tokenizer.json`.
- Tokenizes the local HRM-Text probe text with the real tokenizer.
- Runs the tied Fourier vocab with `vocab_size=65,536`.
- Tests tokenizer-aware orderings:
  - `bpe_rank`
  - `token_frequency`
  - `token_category`
  - `token_category_frequency`
  - `random`

## How To Run

```powershell
rtk python "experiments\Experiment 14 - Tokenizer-Aware Vocab Ordering\tokenizer_aware_vocab_ordering.py" --steps 40 --seeds 1,2,3 --device cuda
```

Locked carry-forward run:

```powershell
rtk python "experiments\Experiment 14 - Tokenizer-Aware Vocab Ordering\tokenizer_aware_vocab_ordering.py" --steps 80 --seeds 1,2,3 --device cuda --orderings bpe_rank,token_frequency
```

Fast smoke:

```powershell
rtk python "experiments\Experiment 14 - Tokenizer-Aware Vocab Ordering\tokenizer_aware_vocab_ordering.py" --steps 10 --seeds 1 --device cuda
```

## Orderings

| Ordering | Meaning |
| --- | --- |
| `bpe_rank` | token ID order from the trained BPE vocab |
| `token_frequency` | most common local tokenizer IDs first |
| `token_category` | special, numeric, word, mixed, punctuation, ASCII-other, Unicode |
| `token_category_frequency` | token category first, then frequency inside each category |
| `random` | deterministic random control |

## Current Read

### Compressed Body Branch

Command:

```powershell
rtk python "experiments\Experiment 14 - Tokenizer-Aware Vocab Ordering\tokenizer_aware_vocab_ordering.py" --steps 40 --seeds 1,2,3 --device cuda
```

Base variant: `fourier-all-tied-fourier-vocab-bias`

| Ordering | Mean Eval Loss | Params | Peak VRAM | Read |
| --- | ---: | ---: | ---: | --- |
| `bpe_rank` | 10.1982 | 131,072 | 243.6 MB | real tokenizer ID baseline |
| `token_frequency` | 7.8712 | 131,072 | 243.6 MB | best result |
| `token_category` | 9.8115 | 131,072 | 243.6 MB | slightly better than BPE rank |
| `token_category_frequency` | 8.4808 | 131,072 | 243.6 MB | strong, but weaker than pure frequency |
| `random` | 11.9928 | 131,072 | 243.6 MB | bad control |

### Vocab-Only Branch

Command:

```powershell
rtk python "experiments\Experiment 14 - Tokenizer-Aware Vocab Ordering\tokenizer_aware_vocab_ordering.py" --steps 40 --seeds 1,2,3 --device cuda --base-variant tied-fourier-vocab-bias
```

Base variant: `tied-fourier-vocab-bias`

| Ordering | Mean Eval Loss | Params | Peak VRAM | Read |
| --- | ---: | ---: | ---: | --- |
| `bpe_rank` | 9.4609 | 458,752 | 247.7 MB | real tokenizer ID baseline |
| `token_frequency` | 7.3996 | 458,752 | 247.7 MB | best result |
| `token_category` | 9.5769 | 458,752 | 247.7 MB | slightly worse than BPE rank |
| `token_category_frequency` | 7.9464 | 458,752 | 247.7 MB | strong, but weaker than pure frequency |
| `random` | 11.0682 | 458,752 | 247.7 MB | bad control |

Read: unlike the byte-token shortcut in Experiment 13, real tokenizer-aware frequency ordering clearly helps in this tiny local run. This is not proof at scale yet, but it is the first ordering result worth carrying forward. The next larger run should lock `token_frequency` against `bpe_rank` and maybe add a real co-occurrence order from Sapient-tokenized data.

## Locked Decision

Carry forward:

```text
ordering = token_frequency
control  = bpe_rank
```

This is locked as the next experimental baseline, not as the final model default. The next run should compare only `bpe_rank` vs `token_frequency` for more steps before changing another part.
