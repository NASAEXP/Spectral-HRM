# Experiment 11 - Vocab Head Ablations

## Goal

Run the seven remaining LM head / embedding ideas as one tracked ablation.

The seven ideas are:

```text
1. dense tied vocab baseline
2. untied Fourier vocab
3. vocab bias
4. embedding scale tuning
5. learned token basis + Fourier hidden basis
6. token ID reordering
7. checkpoint/recompute dense vocab weight
```

## How To Run

```powershell
rtk python "experiments\Experiment 11 - Vocab Head Ablations\vocab_head_ablation.py" --steps 80 --seeds 1,2,3 --device cpu
```

## Notes

This experiment focuses on the vocab/input-output side. It keeps the same local byte-text probe used by Experiments 8-10.

## Result

Command:

```powershell
rtk python "experiments\Experiment 11 - Vocab Head Ablations\vocab_head_ablation.py" --steps 80 --seeds 1,2,3 --device cpu
```

Summary:

| Variant | Mean Eval Loss | Params | Read |
| --- | ---: | ---: | --- |
| dense | 3.7068 | 427,008 | baseline untied dense vocab |
| dense-tied-vocab | 3.5380 | 393,728 | dense weight tying helps a lot |
| tied-fourier-vocab | 3.6070 | 377,088 | Fourier tied works, but trails dense tying |
| tied-fourier-vocab-bias | 3.6020 | 377,348 | tiny improvement from vocab bias |
| tied-fourier-vocab-learned-scale | 3.6070 | 377,089 | no useful change here |
| untied-fourier-vocab | 3.8786 | 393,728 | worse in this probe |
| learned-token-fourier-vocab | 3.5962 | 444,688 | good quality, but more params |
| tied-fourier-vocab-reordered | 3.6297 | 377,088 | reverse token order does not help |
| tied-fourier-vocab-checkpoint | 3.6070 | 377,088 | same quality, useful for memory later |
| fourier-all | 3.6843 | 99,328 | compressed body baseline |
| fourier-all-dense-tied-vocab | 3.4476 | 66,048 | best result in this run |
| fourier-all-tied-fourier-vocab | 3.6896 | 49,408 | very small, but weaker |
| fourier-all-tied-fourier-vocab-bias | 3.6853 | 49,668 | small improvement from bias |
| fourier-all-untied-fourier-vocab | 3.9087 | 66,048 | worse |
| fourier-all-learned-token-fourier-vocab | 3.7029 | 117,008 | not worth params here |

## Read

The best current practical preset is:

```text
fourier-all-dense-tied-vocab
```

That means:

```text
FourierLinear inside the transformer body
+ one normal dense vocab matrix tied between embedding and LM head
```

The tied Fourier vocab branch is still valuable for the extreme-small path. Add bias when using it:

```text
hrm_fourier_all_tied_vocab.yaml now keeps bias enabled.
```

For future VRAM work, `checkpoint_weight` is now implemented and verified. It does not change eval loss, but it can recompute the generated vocab matrix during backward instead of keeping the whole dense temporary alive.
