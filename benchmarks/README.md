# Benchmarks: how well does a configuration restore a known Sun?

`run.py` restores bursts whose truth is known and scores the object and the PSFs against it. Use it before changing a
default, the example, or the filters. A change is only an improvement if it wins on scenes it was not tuned on, by
more than the noise floor, on the same device.

## Run

```bash
python benchmarks/run.py -o a.json                                 # the example's kl.yaml, 128-px patches, stride 32
python benchmarks/run.py -o b.json --patch 64 --border 10          # any change: --loss, --iterations, --config ...
python benchmarks/run.py -o a_s1.json --seed 1                     # the same as a, from a perturbed start
python benchmarks/run.py --compare a.json b.json --floor a_s1.json
```

The default sets need no download. On a Tesla V100S they take about 2 minutes:
- `synth`: 13 synthetic 128-px bursts;
- `aniso-synth`: 12 synthetic 256-px scenes with two-layer seeing.

On the CPU (`--gpu -1`) they take several minutes.

The `real` and `aniso-real` sets are crops of the example's SST/CRISP 8542 Å burst degraded with known seeing. They
need the burst file `spot_20200727_083509_8542_npix512_original.h5`, which is not in the repository: get it as for the
example (`reproducibility/download_script.py`) and pass `--h5 <path>`.

Options:
- `--sets`
- `--patch`, `--stride`, `--border`
- `--loss`, `--optimizer`, `--iterations`
- `--frames`, `--noise`
- `--alt`: height of the upper seeing layer, default 5 km
- `--gpu`, `--seed`
- `--no-control`

See `python benchmarks/run.py -h`.

## Output

One JSON row per scene:

| field | what |
|---|---|
| `err_obj` | rms(object − truth) / std(truth), after sub-pixel alignment, means removed, 16-px border dropped. The truth is seen through the same diffraction mask, patches and recombination as the object. |
| `band_err` | the same per frequency band, 0.05–0.25 / 0.25–0.5 / 0.5–0.75 / 0.75–0.95 of the cutoff. 1 = nothing right there; above 1 = power that is not in the truth. |
| `err_transfer` | R, the error of the transfer function the object actually sees. |
| `err_psf` | per-frame \|OTF\| error. |
| `err_obj_true_psf` | the object from the true PSFs: what a perfect PSF estimate would give. |

`--compare` reports, per set and overall:
- in how many scenes b is better;
- the mean paired change;
- the two-sided sign test p;
- R;
- the scenes more than 25 % worse;
- the band errors;
- with `--floor`, whether the change is beyond the noise floor (the mean |change| between two starts of a).

## Pitfalls

These are measured, and each has cost a wrong conclusion once.

- **Units of the truth.** The library is given frames divided by their own mean. A truth in other units (divided by
  the mean of a larger image, or not divided at all) adds a contrast error of up to 30 % that looks like a real
  difference. `metrics.object_error` raises if the object and truth means differ by more than 3 %.
- **The PSF reference.** An m-px PSF model can only represent the true PSF folded onto an m-px period. A cropped and
  renormalised reference differs from it by 2–10 %. `metrics.fold_psf` does the folding.
- **Which PSF measure.** R predicts the object error (rank correlation +0.71 to +0.88); the per-frame |OTF| error does
  not (−0.18 to −0.47).
- **Devices.** LBFGS amplifies the last bits of the arithmetic:
  - per-scene PSF errors differ by up to 0.24 between a Mac and a Linux CPU;
  - object errors differ by 3–4 % between a V100S and a T4.

  `--compare` refuses runs from different devices. Each device repeats its own results exactly.
- **The noise floor.** Two starts of the same configuration differ by several % per scene (±8.6 % mean |change| on the
  default sets, 64-px patches). A mean change inside it is not a result, however many scenes favour it.
- **The control.** The object from the true PSFs may not be more than 5 % worse than the code's in any scene, and must
  be better over each set. Otherwise the metric or the truth is wrong and the run stops. With large patches the
  code gets within a few % of it, and 1–2 % below it in single scenes: that is allowed.
- **No truth, no ranking.** On real data, none of the quality measures that need no truth was found to pick the better
  configuration:
  - residual χ² overall and at high frequency: the high-frequency χ² is anticorrelated with quality;
  - residual whiteness;
  - Fourier ring correlation between half bursts: the best of them, and still worse than a fixed good configuration.

  This is why the benchmark uses scenes with a known truth.
