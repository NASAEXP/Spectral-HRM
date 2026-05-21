# Experiment 12 - Fourier Vocab Assumption Tests

## Goal

Turn the tied Fourier vocab idea into testable assumptions instead of vibes.

This experiment checks:

- the dense vocab matrix is temporary, not a trainable parameter
- embedding and LM head can reuse the exact same generated matrix
- gradients flow back to the small Fourier coefficient grid
- `checkpoint_weight` can recompute the dense matrix in backward
- DCT/cosine and real FFT-style bases are separate choices
- token ordering can change the result

## What Changed

- Added `basis_type: dct|fft` for Fourier vocab heads.
- Added a real-valued FFT-style basis made from constant, cosine, and sine rows.
- Added focused assumption tests in `tests/test_fourier_vocab_assumptions.py`.
- Added this local runner for small text ablations.

## How To Run

```powershell
rtk python "experiments\Experiment 12 - Fourier Vocab Assumption Tests\fourier_vocab_assumption_tests.py" --steps 80 --seeds 1,2,3 --device cpu
```

For a faster smoke:

```powershell
rtk python "experiments\Experiment 12 - Fourier Vocab Assumption Tests\fourier_vocab_assumption_tests.py" --steps 20 --seeds 1 --device cpu
```

## Variants

| Variant | What it tests |
| --- | --- |
| `tied-fourier-vocab-bias` | DCT/cosine tied vocab baseline |
| `tied-fourier-vocab-bias-fft-basis` | real FFT-style tied vocab basis |
| `tied-fourier-vocab-reordered` | whether token order matters |
| `tied-fourier-vocab-checkpoint` | whether dense W can be recomputed in backward |
| `fourier-all-tied-fourier-vocab-bias` | compressed body plus DCT/cosine vocab |
| `fourier-all-tied-fourier-vocab-bias-fft-basis` | compressed body plus FFT-style vocab |

## Current Read

Command:

```powershell
rtk python "experiments\Experiment 12 - Fourier Vocab Assumption Tests\fourier_vocab_assumption_tests.py" --steps 40 --seeds 1,2,3 --device cuda
```

RTX 3050 Ti local result:

| Variant | Mean Eval Loss | Params | Peak VRAM | Read |
| --- | ---: | ---: | ---: | --- |
| `tied-fourier-vocab-bias` | 3.9079 | 377,348 | 23.8 MB | best tied-vocab-only baseline here |
| `tied-fourier-vocab-bias-fft-basis` | 4.0606 | 377,348 | 23.8 MB | FFT basis works, worse than DCT/cosine here |
| `tied-fourier-vocab-reordered` | 4.0533 | 377,088 | 23.7 MB | token order matters |
| `tied-fourier-vocab-checkpoint` | 3.9118 | 377,088 | 23.7 MB | same quality shape as baseline, recomputes W in backward |
| `fourier-all-tied-fourier-vocab-bias` | 4.0093 | 49,668 | 19.2 MB | tiny compressed body plus vocab still trains |
| `fourier-all-tied-fourier-vocab-bias-fft-basis` | 4.2102 | 49,668 | 19.2 MB | FFT basis trails DCT/cosine in the compressed body too |

Read: the assumptions hold mechanically. The more promising default is still DCT/cosine for now, not the raw FFT-style basis. The checkpoint route is useful for memory work because it preserves behavior while giving us a recompute knob.
